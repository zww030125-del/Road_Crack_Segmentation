"""Joint fine tuning with foreground contrast attenuation, cross-view prediction and locked selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.nn import functional as F
import end_to_end_thin as e
import select_thin_fusion as wf
from train_full_image_baseline import ROOT,FullImages,save_table
from train_pooling_stride_experiment import make_model
from datasets.splits import get_train_val_pairs
from thin_followup import geometry,counts,summarize

OUT=ROOT/'outputs/end_to_end_thin/joint'
BASE=ROOT/'outputs/local_bce_budget/seed_42/positive.pth'
DEVICE=e.DEVICE

def create():
    m=make_model(2,torch.load(BASE,map_location='cpu',weights_only=True))
    for p in m.encoder.parameters():p.requires_grad_(True)
    # Existing train override keeps the pretrained BN running statistics fixed.
    return m.to(DEVICE)


def fit(pairs):
    seed=47;torch.manual_seed(seed);rng=np.random.default_rng(seed)
    m=create()
    opt=torch.optim.AdamW([{'params':m.encoder.parameters(),'lr':.00006},
                           {'params':[p for n,p in m.named_parameters() if not n.startswith('encoder.')],'lr':.00022}])
    steps=1500
    scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[.00006,.00022],total_steps=steps,pct_start=.12)
    items=[]
    for x,y in FullImages(pairs):
        sk,thin,low,_=geometry(x,y[0].numpy())
        items.append((x,y,torch.from_numpy(ndi.maximum_filter(thin,size=3)&y[0].numpy())[None],[np.flatnonzero(z) for z in (low,thin,sk)]))
    m.train()
    for step in range(steps):
        samples=[]
        side=int(rng.choice([112,144,176]))
        for i in range(4):
            x,y,t,pools=items[rng.integers(len(items))]
            h,w=y.shape[-2:];g=rng.choice(4,p=[.3,.25,.3,.15])
            idx=int(rng.choice(pools[g])) if g<3 and len(pools[g]) else int(rng.integers(h*w))
            cy,cx=divmod(idx,w)
            j=int(np.clip(cy-rng.integers(16,side-16),0,h-side))
            k=int(np.clip(cx-rng.integers(16,side-16),0,w-side))
            sample=torch.cat([x[:,j:j+side,k:k+side],y[:,j:j+side,k:k+side],t[:,j:j+side,k:k+side]])
            sample=torch.rot90(sample,int(rng.integers(4)),(-2,-1))
            if rng.random()<.5:sample=sample.flip([-1])
            samples.append(sample)
        data=torch.stack(samples).to(DEVICE)
        x,y,thin=data[:,:3],data[:,3:4],data[:,4:]
        mean=F.avg_pool2d(F.pad(x,(8,8,8,8),mode='reflect'),17,1)
        # Feather the attenuation to avoid mask-edge cues.
        soft=F.avg_pool2d(F.max_pool2d(y,7,1,3),13,stride=1,padding=6)
        fac=float(rng.uniform(.55,1.45))
        if rng.random()<.65:fac=fac*(1-soft*float(rng.uniform(.05,.60)))
        x=(mean+(x-mean)*fac+float(rng.uniform(-.12,.12))).clamp(0,1)
        z=m(x);p=z.sigmoid()
        weight=1+2*thin
        bce=F.binary_cross_entropy_with_logits(z,y,weight=weight,pos_weight=z.new_tensor(3.))
        dice=1-((2*(p*y).sum((1,2,3))+1)/((p+y).sum((1,2,3))+1)).mean()
        loss=bce+.6*dice
        if not torch.isfinite(loss):raise RuntimeError('nonfinite')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();scheduler.step()
        if (step+1)%500==0:print('joint',step+1,flush=True)
    torch.save(m.state_dict(),OUT/'joint.pth')
    return m.eval()


def load():
    old,spec=wf.load()
    m=create();m.load_state_dict(torch.load(OUT/'joint.pth',map_location=DEVICE,weights_only=True))
    width=json.loads((wf.OUT/'locked.json').read_text())['selected']
    return old,spec,m.eval(),width


@torch.inference_mode()
def fields(x,all_models):
    old,spec,m,width=all_models
    raw=wf.representation(x,spec,old)
    base=wf.fuse(raw,width['spec'])
    j=0.
    for k in (0,1,2,3):
        z=m(torch.rot90(x,k,(-2,-1)).contiguous())
        j+=torch.rot90(z,-k,(-2,-1))[0,0].cpu().numpy()/4
    v=torch.rot90(x,1,(-2,-1)).contiguous()
    rotated=e.prior.predict(old[0],old[1],v,[old[2]])[old[2]]
    return raw[0],base,j,np.rot90(rotated,-1).copy(),raw[-1]


def blend(fs,spec):
    orig,base,j,rot,width=fs
    a,r,w=spec
    return (1-a-r)*base+a*j+r*rot-w*width


def evaluate(pairs,models,configs,split):
    meta=[]
    for (x,y),pair in zip(FullImages(pairs),pairs):
        y=y[0].numpy();meta.append((pair[0].name,y,geometry(x,y),fields(x[None].to(DEVICE),models)))
    results=[]
    for spec,t in configs:
        maps=[(fs[0] if spec is None else blend(fs,spec)) for _,_,_,fs in meta]
        if t is None:
            bg=np.concatenate([z[~r[1]] for z,r in zip(maps,meta)])
            i=len(bg)-49000-1;t=float(np.partition(bg,i)[i]);del bg
        rows=[]
        for z,(name,y,g,fs) in zip(maps,meta):
            before=ndi.maximum_filter(fs[0],5)>models[0][3]
            after=ndi.maximum_filter(z,5)>t
            rows.append(dict(image=name,**counts(z,t,y,g),gained=int((~before&after&g[1]).sum()),lost=int((before&~after&g[1]).sum())))
        if split=='test':save_table(OUT/('test_reference.csv' if spec is None else 'test_selected.csv'),rows)
        results.append(dict(spec=spec,threshold=t,**summarize(rows)))
    return results


def main():
    OUT.mkdir(exist_ok=False)
    old,others=wf.load();train,val=get_train_val_pairs(ROOT/'data',seed=42)
    joint=fit(train)
    width=json.loads((wf.OUT/'locked.json').read_text())['selected']
    models=(old,others,joint,width)
    configs=[(None,old[3]),((0.,0.,0.),width['threshold'])]
    configs.extend(((a,r,w),None) for a in (0,.30,.60,.85) for r in (0,.15) for w in (0,1.5,3) if a+r>0)
    rows=evaluate(val,models,configs,'val');ref=rows[0]
    eligible=[z for z in rows[1:] if z['thin']>ref['thin'] and z['low']>ref['low'] and z['long']<ref['long'] and z['dice']>=ref['dice']-.003 and z['fp']<=ref['fp']]
    best=max(eligible,key=lambda z:z['thin_recall']+z['low_recall']) if eligible else ref
    (OUT/'locked.json').write_text(json.dumps(dict(selected=best,validation=rows),indent=2))
    print('locked',best,flush=True)
    if best==ref:return
    pairs=[(p,ROOT/'data/test_lab'/(p.stem+'.png')) for p in sorted((ROOT/'data/test_img').glob('*.jpg'))]
    assert len(pairs)==237
    res=evaluate(pairs,models,[(z['spec'],z['threshold']) for z in (ref,best)],'test')
    (OUT/'test.json').write_text(json.dumps(res,indent=2));print(res,flush=True)


def infer(paths):
    src,dst=paths
    if dst.exists():raise FileExistsError(dst)
    lock=json.loads((OUT/'locked.json').read_text())['selected']
    models=load()
    with Image.open(src) as im:x=torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32)/255).permute(2,0,1).contiguous()[None].to(DEVICE)
    fs=fields(x,models)
    z=fs[0] if lock['spec'] is None else blend(fs,lock['spec'])
    Image.fromarray((z>lock['threshold']).astype('uint8')*255).save(dst,format='PNG')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--infer',nargs=2,type=Path)
    args=parser.parse_args()
    if args.infer:infer(args.infer)
    else:main()
