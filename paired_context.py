"""Matched hard negatives, physical context control, and blinded gap review."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage as nd
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
import bounded_thin_detector as bd
from thin_followup import counts,geometry,summarize
from train_full_image_baseline import ROOT,FullImages,save_table
from datasets.splits import get_train_val_pairs
OUT=ROOT/'outputs/paired_context'
D=bd.f.DEVICE
SEED=83

def encoder(rates):
    layers=[];a=3
    for d in rates:
        layers.extend([nn.Conv2d(a,12,3,padding=d,dilation=d),nn.ReLU()]);a=12
    return nn.Sequential(*layers)

class Detector(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.detail=encoder([1,2,4,4])
        self.context=encoder([1,2,4,20] if kind=='single' else [1,2,4,6] if kind=='dual' else [1,2,4,4])
        self.head=nn.Sequential(nn.Conv2d(24,12,1),nn.ReLU(),nn.Conv2d(12,1,1))
    def forward(self,x):
        h=self.detail(x)
        if self.kind=='dual':
            k=self.context(F.avg_pool2d(x,2,2))
            k=F.interpolate(k,size=h.shape[-2:],mode='bilinear',align_corners=False)
        else:k=self.context(x)
        return self.head(torch.cat([h,k],1))

def descriptor(gray,pts):
    iy,ix=pts.T
    dy,dx=np.gradient(gray);avg=lambda v:nd.uniform_filter(v,9)
    xx,yy,xy=avg(dx*dx),avg(dy*dy),avg(dx*dy)
    den=np.sqrt((xx-yy)**2+4*xy**2)+1e-6
    contrast=avg(gray)-gray
    # Apparent dark-ridge width proxy is shared by both labelled and background patches.
    width=avg((contrast>.015).astype('f'))
    return np.stack([gray[iy,ix]*4,contrast[iy,ix]*6,width[iy,ix],((xx-yy)/den)[iy,ix],(2*xy/den)[iy,ix]],1)

@torch.inference_mode()
def mine(pairs,models,split):
    rng=np.random.default_rng(SEED); images=[];matches=[];gallery={}
    for i,((x,m),pair) in enumerate(zip(FullImages(pairs),pairs)):
        y=m[0].numpy();g=geometry(x,y);xx=x[None].to(D)
        fields=bd.f.fields(xx,models);z=fields[0]
        base=bd.correct(bd.masks(fields,models[0][3]),(.085,.065))
        weak=g[1]&(nd.maximum_filter(models[0][0].net(xx)[0,0].cpu().numpy(),17)<=0)
        images.append((x,m,base,weak))
        gray=x.numpy().mean(0)
        missed=g[0]&~nd.maximum_filter(base,5)
        lab,n=nd.label(missed,np.ones((3,3)));sizes=np.bincount(lab.ravel())
        margin=np.zeros(y.shape,bool);margin[32:-32,32:-32]=True
        dist=nd.distance_transform_edt(~y)
        bg=np.argwhere((dist>=4)&margin)
        if not len(bg):continue
        scores=z[tuple(bg.T)];bg=bg[scores>=np.quantile(scores,.96)]
        if len(bg)>1600:bg=bg[rng.choice(len(bg),1600,False)]
        tree=cKDTree(descriptor(gray,bg))
        pos=np.argwhere(g[1]&missed&margin)
        if len(pos)>30:pos=pos[rng.choice(len(pos),30,False)]
        if split=='train':
            ordinary=np.argwhere(g[1]&margin)
            if len(ordinary):pos=np.vstack([pos,ordinary[rng.choice(len(ordinary),min(12,len(ordinary)),False)]])
        if len(pos):
            _,idx=tree.query(descriptor(gray,pos))
            for pt,neg in zip(pos,bg[idx]):matches.append((i,*pt,*neg))
        # Keep random examples per length/contrast stratum, at most one per image per stratum.
        for a,b,tag in [(2,4,'short'),(5,19,'medium'),(20,100000,'long')]:
            for low in (False,True):
                hit=g[1]&missed&margin&(sizes[lab]>=a)&(sizes[lab]<=b)&(g[2]==low)
                coords=np.argwhere(hit)
                if len(coords):
                    cy,cx=coords[rng.integers(len(coords))]
                    gallery.setdefault((tag,low),[]).append((i,int(cy),int(cx),int(lab[cy,cx])))
    keys=[]
    for (tag,low),pool in gallery.items():
        for k in rng.choice(len(pool),min(2,len(pool)),replace=False):
            i,cy,cx,comp=pool[k];x,m,base,_=images[i];g=geometry(x,m[0].numpy())
            labs,_=nd.label(g[0]&~nd.maximum_filter(base,5),np.ones((3,3)))
            token=hashlib.sha256(f'{split}/{i}/{cy}/{cx}'.encode()).hexdigest()[:10]
            # Jitter target location so the displayed center is not the answer.
            h,w=m.shape[-2:];top=int(np.clip(cy-48+rng.integers(-15,16),0,h-96));left=int(np.clip(cx-48+rng.integers(-15,16),0,w-96))
            sl=np.s_[top:top+96,left:left+96];rgb=(x.numpy().transpose(1,2,0)[sl]*255).round().astype('uint8')
            Image.fromarray(rgb).save(OUT/'blind'/(token+'.png'))
            target=labs[sl]==comp
            Image.fromarray((target*255).astype('uint8')).save(OUT/'key'/(token+'.png'))
            keys.append(dict(id=token,split=split,image=pairs[i][0].name,length=tag,low=low,component=comp,top=top,left=left,visible_points=int(target.sum())))
    save_table(OUT/('key/'+split+'.csv'),keys)
    return images,np.asarray(matches,dtype=int)

def patch(a,y,x):
    h,w=a.shape[-2:];j=int(np.clip(y-48,0,h-96));k=int(np.clip(x-48,0,w-96))
    return a[:,j:j+96,k:k+96],y-j,x-k

def train(data,pairs,kind,ranking):
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED);net=Detector(kind).to(D)
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    for step in range(900):
        xs=[];ys=[];points=[]
        for i,cy,cx,ny,nx in pairs[rng.integers(len(pairs),size=5)]:
            im,m,*_=data[i]
            for y,x in [(cy,cx),(ny,nx)]:
                a,py,px=patch(im,y,x);b,_,_=patch(m,y,x);xs.append(a);ys.append(b);points.append((py,px))
        x=torch.stack(xs).to(D);y=torch.stack(ys).to(D).float()
        pred=net(x);central=np.s_[:,:,32:64,32:64];pc=pred[central];yc=y[central]
        prob=pc.sigmoid()
        loss=F.binary_cross_entropy_with_logits(pc,yc,pos_weight=pc.new_tensor(4.))+.4*(1-(2*(prob*yc).sum()+1)/(prob.sum()+yc.sum()+1))
        if ranking:
            py,px=zip(*points);s=pred[torch.arange(len(x)),0,list(py),list(px)]
            loss=loss+.15*F.softplus(1-s[::2]+s[1::2]).mean()
        opt.zero_grad();loss.backward();opt.step()
    return net.eval()

@torch.inference_mode()
def assess(data,pairs,net):
    zs=[net(x[None].to(D))[0,0].cpu().numpy() for x,*_ in data]
    bg=np.concatenate([z[~row[1][0].numpy()] for z,row in zip(zs,data)])
    budget=sum(int((base&~m[0].numpy()).sum()) for _,m,base,_ in data)
    j=len(bg)-budget-1;t=float(np.partition(bg,j)[j]);rows=[]
    for z,(x,m,base,weak) in zip(zs,data):
        yy=m[0].numpy();r=counts(z,t,yy,geometry(x,yy));r['weak_hits']=int((weak&nd.maximum_filter(z>t,5)).sum());rows.append(r)
    win=np.mean([zs[i][a,b]>zs[i][c,d] for i,a,b,c,d in pairs])
    return dict(threshold=t,pair_accuracy=float(win),**summarize(rows))

def main():
    OUT.mkdir(exist_ok=False);(OUT/'blind').mkdir();(OUT/'key').mkdir()
    models=bd.f.load();tr,va=get_train_val_pairs(ROOT/'data',seed=42)
    data,pairs=mine(tr,models,'train');val,validation_pairs=mine(va,models,'val')
    np.savez(OUT/'pairs.npz',train=pairs,val=validation_pairs)
    result={}
    for kind,rank in [('local',False),('local',True),('single',True),('dual',True)]:
        name=kind+('_rank' if rank else '_ordinary');net=train(data,pairs,kind,rank)
        torch.save(net.state_dict(),OUT/(name+'.pth'));result[name]=assess(val,validation_pairs,net)
        (OUT/'ablation.json').write_text(json.dumps(result,indent=2));print(name,result[name],flush=True)
    refs=[]
    for x,m,base,weak in val:
        r=counts(base,.5,m[0].numpy(),geometry(x,m[0].numpy()));r['weak_hits']=int((weak&nd.maximum_filter(base,5)).sum());refs.append(r)
    result['incumbent']=summarize(refs)
    (OUT/'ablation.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
