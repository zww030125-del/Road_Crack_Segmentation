"""Train a shallow readout, select inference configuration on validation, evaluate locked test thresholds."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.special import expit
from skimage.morphology import skeletonize
import torch
from torch.nn import functional as F
import train_full_image_baseline as b
from datasets.splits import get_train_val_pairs
from train_pooling_stride_experiment import make_model

ROOT=b.ROOT
WEIGHTS=ROOT/'outputs/local_bce_budget/seed_42/positive.pth'
OUT=ROOT/'outputs/thin_multiscale_followup'
BUDGET=52517
OPTIONS=('native','local_contrast','small','large','detail','small_detail','contrast_detail')
SCALES=(0.75,1.,1.5,2.)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def geometry(image, y):
    sk=skeletonize(y)
    thin=sk & (2*ndi.distance_transform_edt(y)<=3)
    low,_=b.thin_low_skeleton(image.numpy(),y)
    region=ndi.maximum_filter(thin,size=17)
    return sk,thin,low,region


def resize(x,size):
    return F.interpolate(x,size=size,mode='bilinear',align_corners=False)


def shallow_features(net,x):
    e=net.encoder
    return e.relu(e.bn1(e.conv1((x-net.mean)/net.std)))


def fit(net,pairs):
    rng=np.random.default_rng(42)
    X=[]; Y=[]
    for image,mask in b.FullImages(pairs):
        truth=mask[0].numpy()
        _,thin,_,_=geometry(image,truth)
        x=image[None].to(device)
        features=shallow_features(net,x)
        logit=net(x)[0,0].cpu().numpy()
        dist=ndi.distance_transform_edt(~truth)
        for subset,target in [(thin,1),(truth,1),((dist>=3)&(dist<=10),0),((dist>2)&(logit>0),0)]:
            ids=np.flatnonzero(subset)
            if not len(ids):continue
            ys,xs=np.unravel_index(rng.choice(ids,min(96,len(ids)),replace=False),truth.shape)
            coords=np.stack(((2*xs+1)/truth.shape[1]-1,(2*ys+1)/truth.shape[0]-1),axis=-1)
            grid=torch.from_numpy(coords.astype('float32')).to(device)[None,None]
            values=F.grid_sample(features,grid,padding_mode='border',align_corners=False)[0,:,0].T.cpu().numpy()
            X.append(values); Y.extend([target]*len(values))
    X=np.concatenate(X).astype('float64'); Y=np.asarray(Y)
    mean=X.mean(0); sd=X.std(0).clip(.001); X=(X-mean)/sd
    def loss(p):
        z=X@p[:-1]+p[-1]
        err=expit(z)-Y
        return np.mean(np.logaddexp(0,z)-Y*z)+.005*p[:-1]@p[:-1],np.r_[X.T@err/len(Y)+.01*p[:-1],err.mean()]
    opt=minimize(loss,np.zeros(X.shape[1]+1),jac=True,method='L-BFGS-B')
    if not opt.success: raise RuntimeError(opt.message)
    w=opt.x[:-1]/sd
    return w,opt.x[-1]-mean@w


def maps(net,x,w,bias,options):
    size=x.shape[-2:]
    native=net(x)
    layers={'native':native}
    if any(k in options for k in ('local_contrast','contrast_detail')):
        mean=F.avg_pool2d(F.pad(x,(10,10,10,10),mode='reflect'),21,stride=1)
        enhanced=(x+1.25*(x-mean)).clamp(0,1)
        layers['local_contrast']=(native+net(enhanced))/2
    if any(k in options for k in ('small','small_detail')):
        layers['small']=(native+resize(net(resize(x,tuple(round(v*.65) for v in size))),size))/2
    if 'large' in options:
        layers['large']=(native+resize(net(resize(x,tuple(round(v*1.75) for v in size))),size))/2
    if any('detail' in k for k in options):
        sh=resize(F.conv2d(shallow_features(net,x),w,bias),size)
        # Also require some crack evidence in the surrounding 17-pixel window.
        context=F.max_pool2d(native,17,stride=1,padding=8)
        bonus=sh.clamp(0,4)*torch.sigmoid(context)
        layers['detail']=native+bonus
        for mode,source in [('small_detail','small'),('contrast_detail','local_contrast')]:
            if mode in options:layers[mode]=layers[source]+bonus
    return {k:layers[k][0,0].cpu().numpy() for k in options}


def counts(score,t,y,g):
    sk,thin,low,region=g
    mask=score>t
    near=ndi.maximum_filter(mask,size=5)
    labels,n=ndi.label(sk&~near,np.ones((3,3)))
    sizes=np.bincount(labels.ravel(),minlength=n+1)
    long=sizes>=20; long[0]=False
    return dict(tp=int((mask&y).sum()),fp=int((mask&~y).sum()),fn=int((~mask&y).sum()),
        thin=int(near[thin].sum()),thin_n=int(thin.sum()),low=int(near[low].sum()),low_n=int(low.sum()),
        long=int(long[labels[thin]].sum()),local_tp=int((region&mask&y).sum()),local_fp=int((region&mask&~y).sum()))


def summarize(rows):
    s={k:sum(r[k] for r in rows) for k in rows[0] if k!='image'}
    s.update(thin_recall=s['thin']/s['thin_n'],low_recall=s['low']/s['low_n'],dice=2*s['tp']/(2*s['tp']+s['fp']+s['fn']),precision=s['tp']/(s['tp']+s['fp']),local_precision=s['local_tp']/(s['local_tp']+s['local_fp']))
    return s


def main():
    import os
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    OUT.mkdir(parents=True,exist_ok=False)
    net=make_model(2,torch.load(WEIGHTS,map_location='cpu',weights_only=True)).to(device).eval()
    train,val=get_train_val_pairs(ROOT/'data',seed=42)
    previous=np.load(ROOT/'outputs/thin_multiscale/shallow.npz')
    w,bias=previous['w'],float(previous['bias'])
    np.savez(OUT/'shallow.npz',w=w,bias=bias)
    w=torch.tensor(w,dtype=torch.float32,device=device)[None,:,None,None]
    bias=torch.tensor([bias],dtype=torch.float32,device=device)
    data=[]; preds={k:[] for k in OPTIONS}
    for (image,mask),pair in zip(b.FullImages(val),val):
        y=mask[0].numpy()
        data.append((pair[0].name,y,geometry(image,y)))
        for k,v in maps(net,image[None].to(device),w,bias,OPTIONS).items():preds[k].append(v)
    summaries={}; thresholds={}
    for k in OPTIONS:
        bg=np.concatenate([v[~d[1]] for v,d in zip(preds[k],data)])
        # Operational candidate keeps 10% validation FP headroom for threshold transfer.
        for tag,budget in [('equal',BUDGET),('locked',BUDGET if k=='native' else int(.98*BUDGET))]:
            t=float(np.partition(bg,len(bg)-budget-1)[len(bg)-budget-1])
            rows=[dict(image=n,**counts(v,t,y,g)) for v,(n,y,g) in zip(preds[k],data)]
            result=summarize(rows)
            b.save_table(OUT/f'val_{k}_{tag}.csv',rows)
            summaries[k+'_'+tag]=dict(threshold=t,**result)
            if tag=='locked':thresholds[k]=t
        print(k,summaries[k+'_locked'],flush=True)
    ref=summaries['native_locked']
    eligible=[k for k in OPTIONS[1:] if all(summaries[k+'_locked'][m]>=ref[m] for m in ('thin_recall','low_recall','dice')) and summaries[k+'_locked']['long']<ref['long']]
    winner=max(eligible,key=lambda k:summaries[k+'_locked']['low_recall']) if eligible else 'native'
    manifest=dict(winner=winner,thresholds=thresholds,validation=summaries,weights=str(WEIGHTS),scales=SCALES,score='logit',validation_fp_budget=BUDGET)
    (OUT/'locked.json').write_text(json.dumps(manifest,indent=2))
    if winner=='native':return
    del preds,data
    test=[(p,ROOT/'data/test_lab'/(p.stem+'.png')) for p in sorted((ROOT/'data/test_img').glob('*.jpg'))]
    assert len(test)==237
    rows={k:[] for k in ('native',winner)}
    for (image,mask),pair in zip(b.FullImages(test),test):
        y=mask[0].numpy(); g=geometry(image,y)
        for k,v in maps(net,image[None].to(device),w,bias,rows).items():
            rows[k].append(dict(image=pair[0].name,**counts(v,thresholds[k],y,g)))
    test_result={}
    for k,r in rows.items():
        b.save_table(OUT/f'test_{k}.csv',r); test_result[k]=summarize(r)
    (OUT/'test.json').write_text(json.dumps(test_result,indent=2))
    print('TEST',test_result,flush=True)

if __name__=='__main__':
    with torch.inference_mode():main()
