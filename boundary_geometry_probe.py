"""Image-disjoint boundary probes on frozen bounded predictions."""
import csv
import json
import numpy as np
import torch
from scipy import ndimage as nd
from PIL import Image,ImageDraw
from skimage.morphology import skeletonize
from torch import nn
import contour_edits as c

OUT=c.ROOT/'outputs/boundary_geometry_probe'
GROUPS={'A':1,'B':6,'C':16}
S=np.ones((3,3),bool)

def features(z,p,im):
    gray=im.numpy().mean(0)
    dist=nd.distance_transform_edt(p)
    band=(dist<=3)&(nd.distance_transform_edt(~p)<=3)
    y,x=np.nonzero(band)
    sk=skeletonize(p)
    if not sk.any(): return band,np.empty((0,16),np.float32)
    idx=nd.distance_transform_edt(~sk,return_distances=False,return_indices=True)
    cy,cx=idx[:,y,x]
    yy,xx=np.indices(p.shape,dtype=float)
    mass=nd.uniform_filter(sk.astype(float),9).clip(1e-9)
    def avg(v): return (nd.uniform_filter(v*sk,9)/mass)[cy,cx]
    a,b=avg(yy),avg(xx)
    angle=.5*np.arctan2(2*(avg(yy*xx)-a*b),avg(xx*xx)-b*b-avg(yy*yy)+a*a)
    ty,tx=np.sin(angle),np.cos(angle)
    sign=np.where((y-cy)*tx-(x-cx)*ty>=0,1,-1)
    ny,nx=tx*sign,-ty*sign
    def at(v,dy,dx,d):return nd.map_coordinates(v,[y+d*dy,x+d*dx],order=1,mode='nearest')
    gy,gx=np.gradient(gray)
    cols=[z[y,x],gray[y,x],gx[y,x],gy[y,x],*( (gray-nd.uniform_filter(gray,w))[y,x] for w in (5,15))]
    cols.extend(at(z,ny,nx,d) for d in (-4,-2,2,4))
    for d in (3,7):
        l,r=at(z,ty,tx,d),at(z,ty,tx,-d)
        cols.extend([(l+r)/2,np.abs(l-r)])
    width=2*dist[tuple(idx)]
    for d in (4,8):
        cols.append((at(width,ty,tx,d)+at(width,ty,tx,-d))/2)
    return band,np.stack(cols,1).astype('f')

def ap(y,s):
    order=np.argsort(-s,kind='stable')
    hits=np.cumsum(y[order]); end=np.r_[np.flatnonzero(np.diff(s[order])),len(s)-1]
    return float(np.sum(np.diff(np.r_[0,hits[end]])*hits[end]/(end+1))/hits[-1])

def fit(x,y,n):
    torch.manual_seed(83)
    v=torch.tensor(x[:,:n]);mu=v.mean(0);sd=v.std(0).clamp_min(1e-5)
    v=(v-mu)/sd;t=torch.tensor(y,dtype=torch.float32)
    net=nn.Sequential(nn.Linear(n,24),nn.ReLU(),nn.Linear(24,1))
    opt=torch.optim.AdamW(net.parameters(),lr=.003)
    for _ in range(1500):
        i=torch.randint(len(v),(1024,))
        loss=nn.functional.binary_cross_entropy_with_logits(net(v[i]).flatten(),t[i])
        opt.zero_grad();loss.backward();opt.step()
    def predict(f):
        with torch.no_grad():return net((torch.from_numpy(f[:,:n])-mu)/sd).flatten().numpy()
    return predict

def transects(y):
    # Common straight, nonjunction sections from existing contour tool.
    sections=c.sections(y,y,skeletonize(y),'')
    rows=[]
    for sec in sections[::4]:
        a,b=sec['y'],sec['x']
        rows.append((a,b))
    if not rows:return np.empty((2,0,49),int)
    sk=skeletonize(y);coords=[]
    for a,b in rows:
        pts=np.array(np.nonzero(sk[a-4:a+5,b-4:b+5])).T
        normal=np.linalg.eigh(np.cov(pts.T))[1][:,0]
        coords.append(np.rint(np.array([[a],[b]])+normal[:,None]*np.arange(-24,25)).astype(int))
    return np.stack(coords,1)

def widths(mask,coords):
    vals=mask[tuple(coords)]
    center=np.argmin(np.where(vals[:,22:27],np.abs(np.arange(-2,3)),9),axis=1)+22
    active=vals[np.arange(len(vals)),center]
    offsets=np.arange(49)
    left=np.where((offsets<center[:,None])&~vals,offsets,-1).max(1)
    right=np.where((offsets>center[:,None])&~vals,offsets,49).min(1)
    return (right-left-1)*active

def main():
    OUT.mkdir(exist_ok=False);rng=np.random.default_rng(873);torch.set_num_threads(2)
    tr,va=c.get_train_val_pairs(c.ROOT/'data',seed=42)
    assert not {a.name for a,b in tr}&{a.name for a,b in va}
    models=c.bd.f.load();xs=[];ys=[];records=[]
    for split,pairs in [('train',tr),('val',va)]:
        for (im,m),(path,_) in zip(c.FullImages(pairs),pairs):
            y=m[0].numpy()
            with torch.inference_mode():fs=c.bd.f.fields(im[None].to(c.device),models)
            p=c.bd.correct(c.bd.masks(fs,models[0][3]),(.085,.065))
            band,f=features(fs[0],p,im)
            if split=='train':
                i=rng.choice(len(f),min(1500,len(f)),replace=False)
                xs.append(f[i]);ys.append(y[band][i])
            else:
                records.append((path.name,y,p,band,f,transects(y),im.numpy().transpose(1,2,0)))
    x,y=np.concatenate(xs),np.concatenate(ys)
    probes={k:fit(x,y,n) for k,n in GROUPS.items()}
    allf=np.concatenate([r[4] for r in records])
    truth=np.concatenate([r[1][r[3]] for r in records])
    edge=np.concatenate([(r[1]&~nd.binary_erosion(r[1],structure=S))[r[3]] for r in records])
    base=np.concatenate([r[2][r[3]] for r in records])
    budget=int((base&~truth).sum())
    valid=edge|~truth
    output={};individual=[];masks={}
    for k in ['base',*GROUPS]:
        if k=='base':chosen=base;area=None
        else:
            score=probes[k](allf)
            area=ap(edge[valid],score[valid])
            order=np.argsort(-score,kind='stable')
            stop=np.flatnonzero(~truth[order])[budget]
            chosen=np.zeros(len(score),bool);chosen[order[:stop]]=True
        pos=tp=fp=fn=err=nwidth=0;masks[k]=[]
        for name,gt,p,band,f,coords,rgb in records:
            pred=p.copy();pred[band]=chosen[pos:pos+len(f)];pos+=len(f)
            masks[k].append(pred)
            a=int((gt&pred).sum());b=int((~gt&pred).sum());d=int((gt&~pred).sum())
            w=np.abs(widths(gt,coords)-widths(pred,coords))
            tp+=a;fp+=b;fn+=d;err+=int(w.sum());nwidth+=len(w)
            individual.append(dict(image=name,group=k,tp=a,fp=b,fn=d,width_sum=int(w.sum()),sections=len(w)))
        output[k]=dict(AP=area,edge_hits=int((edge&chosen).sum()),edges=int(edge.sum()),recall=float(chosen[edge].mean()),width_mae=err/nwidth,sections=nwidth,dice=2*tp/(2*tp+fp+fn),fp=fp)
    # Within-image grouped permutation with the fitted C probe.
    contributions={}
    for tag,sl in [('normal',slice(6,10)),('tangent',slice(10,14)),('width',slice(14,16))]:
        scores=[]
        for r in records:
            f=r[4].copy();f[:,sl]=f[rng.permutation(len(f)),sl];scores.append(probes['C'](f))
        contributions[tag]=output['C']['AP']-ap(edge[valid],np.concatenate(scores)[valid])
    c.save_table(OUT/'per_image.csv',individual)
    changes=[]
    for i,r in enumerate(records):
        gt=r[1];b=masks['B'][i];v=masks['C'][i]
        changes.append(int((b!=gt).sum())-int((v!=gt).sum()))
    order=np.argsort(changes)
    cases=[]
    for label,i in zip(('loss','median','gain'),order[[0,len(order)//2,-1]]):
        name,gt,base,band,f,coords,rgb=records[i]
        sheet=Image.new('RGB',(rgb.shape[1]*3,rgb.shape[0]+24),'white')
        draw=ImageDraw.Draw(sheet)
        for j,(title,mask) in enumerate([('GT',gt),('B',masks['B'][i]),('C',masks['C'][i])]):
            img=(rgb*255).astype('uint8');outline=mask&~nd.binary_erosion(mask,structure=S)
            img[outline]=[255,145,25];sheet.paste(Image.fromarray(img),(j*img.shape[1],24));draw.text((j*img.shape[1]+4,4),title,fill='black')
        sheet.save(OUT/(label+'.png'));cases.append([label,name,changes[i]])
    summary=dict(train_images=len(tr),val_images=len(va),training_pixels=len(x),candidates=len(truth),budget=budget,results=output,permuted_AP_drop=contributions,cases=cases)
    (OUT/'results.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
