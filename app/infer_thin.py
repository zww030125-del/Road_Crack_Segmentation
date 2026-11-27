"""Inference with the validation-locked thin crack model (no ground truth needed)."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.resnet_unet import ResNetUNet

class ThinCrackPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        run=ROOT/'outputs/thin_multiscale_followup'
        lock=json.loads((run/'locked.json').read_text())
        if lock['winner']!='detail':
            raise ValueError('Expected the locked detail configuration')
        self.threshold=lock['thresholds']['detail']
        self.net=ResNetUNet(pretrained=False,freeze_encoder=True)
        self.net.load_state_dict(torch.load(ROOT/'outputs/local_bce_budget/seed_42/positive.pth',map_location='cpu',weights_only=True))
        with np.load(run/'shallow.npz') as fit:
            self.readout=torch.nn.Conv2d(64,1,1)
            with torch.no_grad():
                self.readout.weight.copy_(torch.tensor(fit['w'])[None,:,None,None])
                self.readout.bias.fill_(float(fit['bias']))
        self.eval()

    def forward(self,x):
        net=self.net
        enc=net.encoder
        e0=enc.relu(enc.bn1(enc.conv1((x-net.mean)/net.std)))
        e1=enc.layer1(enc.maxpool(e0))
        e2=enc.layer2(e1)
        e3=enc.layer3(e2)
        d=enc.layer4(e3)
        for decoder,skip in ((net.dec4,e3),(net.dec3,e2),(net.dec2,e1),(net.dec1,e0),(net.dec0,x)):
            d=decoder(net.up_and_concat(d,skip))
        logits=net.output(d)
        shallow=F.interpolate(self.readout(e0),size=x.shape[-2:],mode='bilinear',align_corners=False)
        return logits+shallow.clamp(0,4)*F.max_pool2d(logits,17,1,8).sigmoid()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    if args.output.exists(): p.error('Choose a new output filename')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ThinCrackPredictor().to(device)
    with Image.open(args.image) as im:
        x=torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32)/255).permute(2,0,1).contiguous()
    with torch.inference_mode():
        mask=model(x[None].to(device))[0,0]>model.threshold
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(mask.cpu().numpy().astype('uint8')*255).save(args.output,format='PNG')

if __name__=='__main__':
    main()
