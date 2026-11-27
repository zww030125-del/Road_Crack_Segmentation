"""Cross-section audit and learned local contour edits."""
import json
import numpy as np
import torch
from torch import nn
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import bounded_thin_detector as bd
from thin_followup import counts,summarize,geometry
from train_full_image_baseline import ROOT,FullImages,save_table
from datasets.splits import get_train_val_pairs
OUT=ROOT/'outputs/contour_edits'
device=bd.f.DEVICE

@torch.inference_mode()
def inputs(im,models):
    xx=im[None].to(device)
    fs=bd.f.fields(xx,models)
    p=bd.correct(bd.masks(fs,models[0][3]),(.085,.065))
    base=models[0][0].net
    enc=base.encoder
    feat=enc.relu(enc.bn1(enc.conv1((xx-base.mean)/base.std)))
    # Summarize the frozen stem in eight channel groups on the original pixel grid.
    feat=feat.reshape(1,8,8,*feat.shape[-2:]).amax(2)
    feat=torch.nn.functional.interpolate(feat,size=p.shape,mode='bilinear',align_corners=False)[0].cpu().numpy()
    dist=ndi.distance_transform_edt(p)-ndi.distance_transform_edt(~p)
    gray=im.numpy().mean(0)
    x=np.concatenate([im.numpy(),feat/4,np.stack([np.clip(v,-12,12)/4 for v in fs[:4]]),
                      np.stack([p,dist.clip(-8,8)/4,(gray-ndi.uniform_filter(gray,15))*8])])
    return x.astype('float32'),p,abs(dist)<=4

def sections(y,p,sk,name):
    degree=ndi.convolve(sk.astype(int),np.ones((3,3),int))-1
    good=sk & ~ndi.maximum_filter(sk&(degree!=2),11)
    yy,xx=np.nonzero(good);rows=[];offset=np.arange(-24,25)
    for cy,cx in zip(yy,xx):
        if min(cy,cx,y.shape[0]-1-cy,y.shape[1]-1-cx)<25:continue
        pts=np.array(np.nonzero(sk[cy-4:cy+5,cx-4:cx+5])).T
        eig,v=np.linalg.eigh(np.cov(pts.T))
        if eig[1]<5*max(.01,eig[0]):continue
        n=v[:,0];n=n if n[1]>=0 else -n
        iy=np.rint(cy+n[0]*offset).astype(int);ix=np.rint(cx+n[1]*offset).astype(int)
        spans=[]
        for mask in (y,p):
            active=mask[iy,ix];ids=np.flatnonzero(active & (abs(offset)<=2))
            if not len(ids):break
            i=ids[np.argmin(abs(offset[ids]))];lo=hi=i
            while lo>0 and active[lo-1]:lo-=1
            while hi<48 and active[hi+1]:hi+=1
            if lo==0 or hi==48:break
            spans.append((offset[lo],offset[hi]))
        if len(spans)!=2:continue
        (l,r),(a,b)=spans
        left=int(a-l);right=int(r-b)
        kind=('narrow' if min(left,right)>=0 and left+right>0 else 'wide' if max(left,right)<=0 and left+right<0 else 'shift' if left*right<0 else 'equal')
        rows.append(dict(image=name,y=cy,x=cx,left_in=left,right_in=right,width_deficit=left+right,offset=(left-right)/2,kind=kind))
    return rows

def model():
    layers=[]
    for a,b,d in [(18,24,1),(24,24,2),(24,24,3),(24,16,1)]:
        layers+=[nn.Conv2d(a,b,3,padding=d,dilation=d),nn.SiLU()]
    return nn.Sequential(*layers,nn.Conv2d(16,1,1)).to(device)

def fit(pairs,models):
    rng=np.random.default_rng(79);torch.manual_seed(79);patches=[]
    for im,m in FullImages(pairs):
        x,p,band=inputs(im,models);y=m[0].numpy()
        for region in [band&(p!=y),band&(p==y),band]:
            iy,ix=np.nonzero(region)
            if not len(iy):continue
            for i in rng.choice(len(iy),8):
                j=np.clip(iy[i]-16,0,y.shape[0]-32);k=np.clip(ix[i]-16,0,y.shape[1]-32)
                sl=np.s_[j:j+32,k:k+32]
                patches.append((x[:,sl[0],sl[1]].astype('float16'),y[sl].copy(),band[sl].copy()))
    net=model();opt=torch.optim.AdamW(net.parameters(),lr=.0008)
    for step in range(1600):
        batch=[patches[i] for i in rng.integers(len(patches),size=24)]
        x,y,b=[torch.tensor(np.stack(z),device=device).float() for z in zip(*batch)]
        logits=net(x)[:,0]
        loss=nn.functional.binary_cross_entropy_with_logits(logits[b.bool()],y[b.bool()])
        opt.zero_grad();loss.backward();opt.step()
    torch.save(net.state_dict(),OUT/'head.pth');return net.eval()

def collect(pairs,models):
    return [(pair[0].name,im,m[0].numpy(),geometry(im,m[0].numpy())) for pair,(im,m) in zip(pairs,FullImages(pairs))]

def stats(items,c):
    rows=[]
    for name,y,g,p,band,z in items:
        if c=='keep':q=p
        elif c=='dilate':q=ndi.maximum_filter(p,3)
        else:q=(p&~(band&(z<c[0]))) | (~p&band&(z>c[1]))
        rows.append(dict(image=name,**counts(q,.5,y,g),recovered=int((~p&q&y).sum()),added_fp=int((~p&q&~y).sum()),removed_tp=int((p&~q&y).sum()),removed_fp=int((p&~q&~y).sum())))
    return dict(config=c,**summarize(rows)),rows

@torch.inference_mode()
def prepare(data,models,net):
    items=[]
    for n,im,y,g in data:
        x,p,band=inputs(im,models)
        z=net(torch.from_numpy(x)[None].to(device))[0,0].cpu().numpy()
        items.append((n,y,g,p,band,z))
    return items

def main():
    OUT.mkdir(exist_ok=False);models=bd.f.load();tr,val=get_train_val_pairs(ROOT/'data',seed=42)
    data=collect(val,models);sec=[];areas=[]
    for n,im,y,g in data:
        _,p,_=inputs(im,models);part=sections(y,p,g[0],n);sec.extend(part)
        # Attribute FN to the nearest GT skeleton point only when that exact point was audited.
        if g[0].any():
            nearest=tuple(ndi.distance_transform_edt(~g[0],return_distances=False,return_indices=True))
            labels=np.zeros_like(y,dtype=np.uint8)
            names=['other','narrow','wide','shift','equal']
            for r in part:labels[r['y'],r['x']]=names.index(r['kind'])
            vals=labels[nearest][y&~p]
            areas.append(dict(image=n,**{k:int((vals==i).sum()) for i,k in enumerate(names)}))
    save_table(OUT/'sections.csv',sec);save_table(OUT/'area.csv',areas)
    print('sections',len(sec),{k:sum(r[k] for r in areas) for k in names},flush=True)
    net=fit(tr,models);items=prepare(data,models,net)
    ref,_=stats(items,'keep');dil,_=stats(items,'dilate')
    options=[stats(items,(drop,add))[0] for drop in (-4,-2,-1,0,.5) for add in (0,.5,1,1.5,2,3)]
    eligible=[r for r in options if all(r[k]>=ref[k] for k in ('thin','low','local_precision')) and r['long']<=ref['long'] and r['fp']<=ref['fp']]
    best=max([ref]+eligible,key=lambda r:r['dice'])
    (OUT/'validation.json').write_text(json.dumps(dict(reference=ref,dilation=dil,selected=best,grid=options),indent=2))
    print('selected',best,flush=True)
    if best==ref:return
    pairs=[(p,ROOT/'data/test_lab'/(p.stem+'.png')) for p in sorted((ROOT/'data/test_img').glob('*.jpg'))]
    assert len(pairs)==237
    items=prepare(collect(pairs,models),models,net);result=[]
    for tag,c in [('keep','keep'),('dilate','dilate'),('learned',best['config'])]:
        res,rows=stats(items,c);result.append(res);save_table(OUT/(tag+'.csv'),rows)
    (OUT/'test.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()
