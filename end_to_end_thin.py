"""End-to-end fine-crack specialist and validation-locked fusion with the previous detector."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch import nn
from torch.nn import functional as F
import weak_segment_final as prior
from train_full_image_baseline import ROOT, FullImages, save_table
from datasets.splits import get_train_val_pairs
from thin_followup import geometry, counts, summarize

OUT=ROOT/'outputs/end_to_end_thin'
DEVICE=prior.DEVICE


def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.SiLU())


class Specialist(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.ModuleList([block(a,b) for a,b in [(5,16),(16,24),(24,40),(40,64)]])
        self.dec=nn.ModuleList([block(a,b) for a,b in [(104,40),(64,24),(40,16)]])
        self.out=nn.Conv2d(16,2,1)

    def forward(self,x):
        gray=(x*x.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        avg=F.avg_pool2d(F.pad(gray,(8,8,8,8),mode='replicate'),17,1)
        scale=F.avg_pool2d(F.pad((gray-avg).abs(),(8,8,8,8),mode='replicate'),17,1)
        h=torch.cat([2*x-1,(gray-avg)*8,((gray-avg)/(scale+.025)).clamp(-4,4)],1)
        skips=[]
        for i,enc in enumerate(self.enc):
            h=enc(F.max_pool2d(h,2,ceil_mode=True) if i else h);skips.append(h)
        for dec,s in zip(self.dec,skips[-2::-1]):
            h=dec(torch.cat([F.interpolate(h,size=s.shape[-2:],mode='bilinear',align_corners=False),s],1))
        return self.out(h)


def train(pairs,seed):
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    net=Specialist().to(DEVICE)
    items=[]
    for image,mask in FullImages(pairs):
        y=mask[0].numpy()
        sk,thin,low,_=geometry(image,y)
        items.append((image,mask,torch.from_numpy(ndi.maximum_filter(sk,size=3)&y)[None],
                      [np.flatnonzero(p) for p in [low,thin,sk]]))
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    steps=1800
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.0015,total_steps=steps,pct_start=.10)
    for step in range(steps):
        xs=[];ys=[];ss=[]
        side=112
        for _ in range(8):
            x,y,s,pools=items[rng.integers(len(items))]
            h,w=y.shape[-2:]
            g=rng.choice(4,p=[.25,.22,.33,.20])
            loc=int(rng.choice(pools[g])) if g<3 and len(pools[g]) else int(rng.integers(h*w))
            cy,cx=divmod(loc,w)
            top=int(np.clip(cy-rng.integers(20,side-20),0,h-side))
            left=int(np.clip(cx-rng.integers(20,side-20),0,w-side))
            sample=torch.cat([x,y,s])[:,top:top+side,left:left+side]
            sample=torch.rot90(sample,int(rng.integers(4)),(-2,-1))
            if rng.random()<.5:sample=sample.flip([-1])
            xs.append(sample[:3]);ys.append(sample[3:4]);ss.append(sample[4:])
        x=torch.stack(xs).to(DEVICE);y=torch.stack(ys).to(DEVICE);s=torch.stack(ss).to(DEVICE)
        # Reduce local contrast with no change in label geometry.
        avg=F.avg_pool2d(F.pad(x,(6,6,6,6),mode='reflect'),13,1)
        factor=torch.from_numpy(rng.uniform(.42,1.55,(len(x),1,1,1)).astype('float32')).to(DEVICE)
        x=(avg+(x-avg)*factor+float(rng.uniform(-.10,.10))).clamp(0,1)
        z=net(x)
        def criterion(a,target):
            p=a.sigmoid()
            return F.binary_cross_entropy_with_logits(a,target,pos_weight=a.new_tensor(3.))+1-((2*(p*target).sum((1,2,3))+1)/((p+target).sum((1,2,3))+1)).mean()
        loss=criterion(z[:,:1],y)+.35*criterion(z[:,1:],s)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();sched.step()
        if (step+1)%600==0:print('trained',seed,step+1,flush=True)
    torch.save(net.state_dict(),OUT/f'specialist_{seed}.pth')
    return net.eval()


def previous():
    lock=json.loads((prior.OUT/'selection.json').read_text())
    key=lock['winner'];kind=key.split('_')[0]
    m=prior.Readout(kind=='spatial').to(DEVICE)
    m.load_state_dict(torch.load(prior.OUT/(kind+'.pth'),map_location=DEVICE,weights_only=True))
    return prior.setup(),{kind:m.eval()},key,lock['threshold']


@torch.inference_mode()
def fields(x,networks,old):
    base,heads,key,_=old
    original=prior.predict(base,heads,x,[key])[key]
    ensemble=[]
    for model in networks:
        a=model(x);b=model(x.flip([-1])).flip([-1])
        ensemble.append(((a+b)/2)[0].cpu().numpy())
    logits=np.mean(ensemble,axis=0)
    return original,logits[0],logits[1]


def combine(maps,alpha,aux):
    a,b,c=maps
    return (1-alpha)*a+alpha*(b+aux*c)


def evaluate(pairs,models,old,val,lock=None):
    candidates=[(0.,0.)]+[(a,b) for a in (.25,.5,.75,1.) for b in (0.,.35,1.)] if val else [(0.,0.),tuple(lock['mix'])]
    records=[];meta=[]
    for (x,y),pair in zip(FullImages(pairs),pairs):
        y=y[0].numpy();fields_i=fields(x[None].to(DEVICE),models,old)
        records.append(fields_i)
        meta.append((pair[0].name,y,geometry(x,y)))
    results=[]
    for a,b in dict.fromkeys(candidates):
        zs=[combine(fs,a,b) for fs in records]
        if a==0:t=old[3]
        elif val:
            bg=np.concatenate([z[~r[1]] for z,r in zip(zs,meta)])
            j=len(bg)-49000-1;t=float(np.partition(bg,j)[j]);del bg
        else:t=lock['threshold']
        rows=[]
        for z,(n,y,g),maps in zip(zs,meta,records):
            r=counts(z,t,y,g)
            before=ndi.maximum_filter(maps[0],size=5)>old[3]
            after=ndi.maximum_filter(z,size=5)>t
            rows.append(dict(image=n,**r,gained=int((g[1]&~before&after).sum()),lost=int((g[1]&before&~after).sum())))
        tag='val' if val else 'test'
        save_table(OUT/f'{tag}_{a}_{b}.csv',rows)
        results.append(dict(mix=[a,b],threshold=t,**summarize(rows)))
    return results


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    old=previous()
    train_pairs,val=get_train_val_pairs(ROOT/'data',seed=42)
    models=[train(train_pairs,seed) for seed in (42,43)]
    rows=evaluate(val,models,old,True);ref=rows[0]
    eligible=[r for r in rows[1:] if r['thin']>ref['thin'] and r['low']>ref['low'] and r['long']<ref['long'] and r['dice']>=ref['dice']-.003]
    best=max(eligible,key=lambda r:r['low_recall']+r['thin_recall']) if eligible else ref
    (OUT/'selection.json').write_text(json.dumps(dict(selected=best,validation=rows),indent=2))
    print('selected',best,flush=True)
    if best==ref:return
    pairs=[(p,ROOT/'data/test_lab'/(p.stem+'.png')) for p in sorted((ROOT/'data/test_img').glob('*.jpg'))]
    assert len(pairs)==237
    result=evaluate(pairs,models,old,False,best)
    (OUT/'test.json').write_text(json.dumps(result,indent=2))
    print('test',result,flush=True)


def infer(paths):
    image,output=paths
    if output.exists():raise FileExistsError(output)
    best=json.loads((OUT/'selection.json').read_text())['selected']
    old=previous();models=[]
    for seed in (42,43):
        m=Specialist().to(DEVICE);m.load_state_dict(torch.load(OUT/f'specialist_{seed}.pth',map_location=DEVICE,weights_only=True));models.append(m.eval())
    with Image.open(image) as source:x=torch.from_numpy(np.asarray(source.convert('RGB'),dtype=np.float32)/255).permute(2,0,1).contiguous()[None].to(DEVICE)
    z=combine(fields(x,models,old),*best['mix'])
    Image.fromarray((z>best['threshold']).astype('uint8')*255).save(output,format='PNG')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--infer',nargs=2,type=Path)
    args=p.parse_args()
    if args.infer:infer(args.infer)
    else:run()
