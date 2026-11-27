"""Weak-segment mining, paired readouts, validation selection, and locked inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from scipy import ndimage as ndi
from app.infer_thin import ThinCrackPredictor
from train_full_image_baseline import ROOT, FullImages, save_table
from datasets.splits import get_train_val_pairs
from thin_followup import geometry, counts, summarize

OUT=ROOT/'outputs/weak_segment_final'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS=18
PATCH=80


class Readout(nn.Module):
    def __init__(self, spatial):
        super().__init__()
        layers=[nn.Conv2d(67,24,1),nn.ReLU()]
        for dilation in (1,2,4,1):
            k=3 if spatial else 1
            layers.extend([nn.Conv2d(24,24,k,padding=dilation if spatial else 0,dilation=dilation),nn.ReLU()])
        self.layers=nn.Sequential(*layers,nn.Conv2d(24,1,1))

    def forward(self, base, x):
        net=base.net
        enc=net.encoder
        with torch.no_grad():
            e=enc.relu(enc.bn1(enc.conv1((x-net.mean)/net.std)))
            avg=F.avg_pool2d(F.pad(x,(8,8,8,8),mode='reflect'),17,stride=1)
            detail=F.interpolate((x-avg)*6,size=e.shape[-2:],mode='bilinear',align_corners=False)
        return F.interpolate(self.layers(torch.cat([e,detail],1)),size=x.shape[-2:],mode='bilinear',align_corners=False)


def setup():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    return ThinCrackPredictor().eval().to(DEVICE)


@torch.inference_mode()
def mine(base,pairs):
    items=[]; totals=np.zeros(4,dtype=np.int64)
    for image,mask in FullImages(pairs):
        y=mask[0].numpy()
        sk,thin,_,_=geometry(image,y)
        z=base(image[None].to(DEVICE))[0,0].cpu().numpy()
        missing=sk & (ndi.maximum_filter(z,5)<=base.threshold)
        lab,n=ndi.label(missing,np.ones((3,3)))
        sizes=np.bincount(lab.ravel(),minlength=n+1);sizes[0]=0
        long=thin & (sizes[lab]>=20)
        pools=[np.flatnonzero(m) for m in (long,thin & missing,thin,(~y)&(ndi.distance_transform_edt(~y)>=3)&(z>0))]
        items.append((image,mask,pools))
        totals+=[len(p) for p in pools]
    print('training pixels in mining pools (overlap possible)',totals.tolist(),flush=True)
    return items


def train_readout(base,items,spatial):
    torch.manual_seed(42)
    rng=np.random.default_rng(42)
    model=Readout(spatial).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    for epoch in range(EPOCHS):
        order=rng.permutation(np.repeat(np.arange(len(items)),4))
        for start in range(0,len(order),16):
            xs=[]; ys=[]
            for index in order[start:start+16]:
                x,y,pools=items[index]
                h,w=y.shape[-2:]
                p=pools[rng.choice(4,p=[.32,.16,.30,.22])]
                loc=int(rng.choice(p)) if len(p) else int(rng.integers(h*w))
                cy,cx=divmod(loc,w)
                top=int(np.clip(cy-PATCH//2+rng.integers(-15,16),0,h-PATCH))
                left=int(np.clip(cx-PATCH//2+rng.integers(-15,16),0,w-PATCH))
                x=x[:,top:top+PATCH,left:left+PATCH]; y=y[:,top:top+PATCH,left:left+PATCH]
                if rng.random()<.5:x=x.flip([-1]);y=y.flip([-1])
                xs.append(x);ys.append(y)
            z=model(base,torch.stack(xs).to(DEVICE)); truth=torch.stack(ys).float().to(DEVICE)
            prob=z.sigmoid()
            loss=F.binary_cross_entropy_with_logits(z,truth,pos_weight=z.new_tensor(4.))+0.5*(1-(2*(prob*truth).sum()+1)/(prob.sum()+truth.sum()+1))
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if epoch in (0,8,17):print('spatial',spatial,'epoch',epoch+1,flush=True)
    return model.eval()


def candidate_names():
    return ['reference','contrast']+[f'{kind}_{offset}_{context}' for kind in ('point','spatial') for offset in (-1,0,1) for context in (0,1)]


@torch.inference_mode()
def predict(base,heads,x,names):
    original=base(x)
    if any(k=='contrast' or k.endswith('_1') for k in names):
        avg=F.avg_pool2d(F.pad(x,(10,10,10,10),mode='reflect'),21,stride=1)
        contrast=.70*original+.30*base((x+1.25*(x-avg)).clamp(0,1))
    sides={kind:m(base,x) for kind,m in heads.items() if any(k.startswith(kind+'_') for k in names)}
    result={}
    for k in names:
        if k=='reference':z=original
        elif k=='contrast':z=contrast
        else:
            kind,offset,context=k.split('_')
            z=torch.logaddexp(contrast if context=='1' else original,sides[kind]+int(offset))
        result[k]=z[0,0].cpu().numpy()
    return result


def row_for(name,z,t,truth,g,weak,before):
    near=ndi.maximum_filter(z,5)>t
    return dict(image=name,**counts(z,t,truth,g),weak_hits=int((weak&near).sum()),weak_n=int(weak.sum()),
                gained=int((g[1]&~before&near).sum()),lost=int((g[1]&before&~near).sum()))


def experiment():
    OUT.mkdir(parents=True,exist_ok=False)
    base=setup()
    train,val=get_train_val_pairs(ROOT/'data',seed=42)
    items=mine(base,train)
    heads={}
    for kind in ('point','spatial'):
        heads[kind]=train_readout(base,items,kind=='spatial')
        torch.save(heads[kind].state_dict(),OUT/(kind+'.pth'))
    del items
    names=candidate_names()
    # Cache validation maps on disk to bound peak RAM.
    import tempfile
    summaries={}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        metas=[]
        for i,((x,y),pair) in enumerate(zip(FullImages(val),val)):
            truth=y[0].numpy();g=geometry(x,truth)
            xx=x[None].to(DEVICE)
            with torch.inference_mode(): native=base.net(xx)[0,0].cpu().numpy()
            preds=predict(base,heads,xx,names)
            before=ndi.maximum_filter(preds['reference'],5)>base.threshold
            metas.append((pair[0].name,truth,g,g[1]&(ndi.maximum_filter(native,17)<=0),before))
            for k,z in preds.items():np.save(Path(tmp)/f'{i}_{k}.npy',z)
        for k in names:
            scores=[np.load(Path(tmp)/f'{i}_{k}.npy',mmap_mode='r') for i in range(len(val))]
            if k=='reference':t=base.threshold
            else:
                bg=np.concatenate([z[~meta[1]] for z,meta in zip(scores,metas)])
                idx=len(bg)-50000-1;t=float(np.partition(bg,idx)[idx]);del bg
            rows=[row_for(n,z,t,y,g,weak,before) for z,(n,y,g,weak,before) in zip(scores,metas)]
            summaries[k]=dict(threshold=t,**summarize(rows))
            save_table(OUT/f'val_{k}.csv',rows)
            del scores
    ref=summaries['reference']
    eligible=[k for k,s in summaries.items() if s['low']>ref['low'] and s['thin']>ref['thin'] and s['long']<ref['long'] and s['weak_hits']>ref['weak_hits'] and s['fp']<=ref['fp'] and s['dice']>=ref['dice']-.003]
    winner=max(eligible,key=lambda k:(summaries[k]['weak_hits'],summaries[k]['low'])) if eligible else 'reference'
    lock=dict(winner=winner,threshold=summaries[winner]['threshold'],validation=summaries)
    (OUT/'selection.json').write_text(json.dumps(lock,indent=2))
    print('selected',winner,summaries[winner],flush=True)
    if winner=='reference':return
    test=[(p,ROOT/'data/test_lab'/(p.stem+'.png')) for p in sorted((ROOT/'data/test_img').glob('*.jpg'))]
    assert len(test)==237
    rows={k:[] for k in ('reference',winner)}
    for (x,y),pair in zip(FullImages(test),test):
        truth=y[0].numpy();g=geometry(x,truth);xx=x[None].to(DEVICE)
        with torch.inference_mode(): native=base.net(xx)[0,0].cpu().numpy()
        preds=predict(base,heads,xx,rows)
        before=ndi.maximum_filter(preds['reference'],5)>base.threshold
        weak=g[1]&(ndi.maximum_filter(native,17)<=0)
        for k,z in preds.items():rows[k].append(row_for(pair[0].name,z,summaries[k]['threshold'],truth,g,weak,before))
    result={k:summarize(rr) for k,rr in rows.items()}
    for k,rr in rows.items():save_table(OUT/f'test_{k}.csv',rr)
    (OUT/'test.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)


def infer(image,output):
    lock=json.loads((OUT/'selection.json').read_text())
    base=setup();heads={};key=lock['winner'];kind=key.split('_')[0]
    if kind in ('point','spatial'):
        m=Readout(kind=='spatial').to(DEVICE)
        m.load_state_dict(torch.load(OUT/(kind+'.pth'),map_location=DEVICE,weights_only=True));heads[kind]=m.eval()
    with Image.open(image) as im:x=torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32)/255).permute(2,0,1).contiguous()[None].to(DEVICE)
    mask=predict(base,heads,x,[key])[key]>lock['threshold']
    if output.exists():raise FileExistsError(output)
    Image.fromarray(mask.astype('uint8')*255).save(output,format='PNG')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infer',nargs=2,type=Path,metavar=('IMAGE','OUTPUT'))
    args=parser.parse_args()
    if args.infer:infer(*args.infer)
    else:experiment()
