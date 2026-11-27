"""Native and tiled DeepCrack inference on the same 60 validation images."""
import csv,json,time,tempfile,gc,platform
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as nd
from skimage.morphology import skeletonize
import bounded_thin_detector as bd
from thin_followup import counts,geometry,summarize

OUT=bd.ROOT/'outputs/field_inference_audit'
D=bd.f.DEVICE
CORE=192
S=np.ones((3,3),bool)
CONFIG=json.loads((bd.OUT/'locked.json').read_text())['selected']['config']

def tensor(rgb):
    return torch.from_numpy(np.array(rgb,dtype='float32')/255).permute(2,0,1).contiguous()[None].to(D)

def scores(rgb,models,full):
    x=tensor(rgb)
    if full:return bd.f.fields(x,models)[:3]
    base,heads,key,_=models
    return (bd.f.e.prior.predict(base,heads,x,[key])[key],)

def decide(f,models,full):
    threshold=(models[0] if full else models)[3]
    return bd.correct(bd.masks(f,threshold),CONFIG) if full else f[0]>threshold

def tiles(rgb,models,full,offset,halo):
    h,w=rgb.shape[:2]
    bottom=(-(h+offset))%CORE;right=(-(w+offset))%CORE
    padded=np.pad(rgb,((offset+halo,bottom+halo),(offset+halo,right+halo),(0,0)),mode='reflect')
    fields=None
    for y in range(0,h+offset+bottom,CORE):
        for x in range(0,w+offset+right,CORE):
            tile=padded[y:y+CORE+2*halo,x:x+CORE+2*halo]
            f=scores(tile,models,full)
            if fields is None:fields=[np.empty((h+offset+bottom,w+offset+right),'f') for _ in f]
            for plane,part in zip(fields,f):
                plane[y:y+CORE,x:x+CORE]=part[halo:halo+CORE,halo:halo+CORE]
    return decide([a[offset:offset+h,offset:offset+w] for a in fields],models,full)

def distance(shape,offset):
    yy,xx=np.indices(shape)
    def dist(z):
        k=(z+offset)%CORE
        d=np.minimum(k,CORE-1-k)
        return d
    return np.minimum(dist(yy),dist(xx))

def review(pred,y,g):
    labels,n=nd.label(pred,S)
    close=nd.maximum_filter(y,5)
    hits=np.unique(labels[close]);fake=np.ones(n+1,bool);fake[hits]=False;fake[0]=False
    pieces,count=nd.label(g[0]&~nd.maximum_filter(pred,5),S)
    sizes=np.bincount(pieces.ravel(),minlength=count+1)[1:]
    return dict(regions=n,false_regions=int(fake.sum()),long_gaps=int((sizes>=20).sum()))

def breaks(near,inside,sk):
    total=0
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        a=(slice(0,-1) if dy else slice(None),slice(0,-1) if dx==1 else slice(1,None) if dx==-1 else slice(None))
        b=(slice(1,None) if dy else slice(None),slice(1,None) if dx==1 else slice(0,-1) if dx==-1 else slice(None))
        total+=int((inside[a]&inside[b]&sk[a]&sk[b]&(near[a]!=near[b])).sum())
    return total

def timed(fn,path,dst):
    torch.cuda.synchronize();start=time.perf_counter()
    with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
    pred=fn(rgb)
    Image.fromarray(pred.astype('uint8')*255).save(dst)
    torch.cuda.synchronize()
    return pred,(time.perf_counter()-start)*1000

@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=False)
    _,pairs=bd.get_train_val_pairs(bd.ROOT/'data',seed=42)
    totals={};detail=[];seams=[]
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)/'prediction.png'
        for full,tag in [(False,'primary'),(True,'full')]:
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            models=bd.f.load() if full else bd.f.e.previous()
            def native(rgb):return decide(scores(rgb,models,full),models,full)
            for _ in range(3):timed(native,pairs[0][0],dest)
            mem=0;times=[];rs=[];reviewed=[]
            for path,lab in pairs:
                pred,ms=timed(native,path,dest);times.append(ms)
                mem=max(mem,torch.cuda.max_memory_allocated())
                with Image.open(path) as im:x=tensor(im)[0].cpu()
                with Image.open(lab) as im:y=np.asarray(im.convert('L'))>127
                g=geometry(x,y);r=counts(pred,.5,y,g);v=review(pred,y,g)
                rs.append(r);reviewed.append(v)
                detail.append(dict(model=tag,image=path.name,ms=ms,**r,**v))
            totals[tag]=dict(**summarize(rs),p50=float(np.median(times)),p95=float(np.percentile(times,95)),gpu_allocated_peak_MiB=mem/2**20,**{k:sum(r[k] for r in reviewed) for k in reviewed[0]})
            del models
            print(tag,totals[tag],flush=True)
        # Prefer higher low-contrast skeleton coverage under the native primary FP count.
        selected=max((k for k in totals if totals[k]['fp']<=totals['primary']['fp']),key=lambda k:totals[k]['low_recall'])
        full=selected=='full';models=bd.f.load() if full else bd.f.e.previous()
        for offset in (0,96):
            for halo in (0,32):timed(lambda rgb:tiles(rgb,models,full,offset,halo),pairs[0][0],dest)
        for path,lab in pairs:
            with Image.open(path) as im:x=tensor(im)[0].cpu()
            with Image.open(lab) as im:y=np.asarray(im.convert('L'))>127
            sk,thin,_,_=geometry(x,y)
            ds=[distance(y.shape,k) for k in (0,96)]
            interior=np.zeros(y.shape,bool);interior[32:-32,32:-32]=True
            for halo in (0,32):
                masks=[];times=[]
                for offset in (0,96):
                    pred,ms=timed(lambda rgb:tiles(rgb,models,full,offset,halo),path,dest)
                    masks.append(nd.maximum_filter(pred,5));times.append(ms)
                for j in (0,1):
                    same=interior&(ds[j]<=8)&(ds[1-j]>=48)
                    target=same&thin
                    seams.append(dict(image=path.name,halo=halo,offset=j*96,ms=times[j],points=int(target.sum()),edge_hits=int((target&masks[j]).sum()),center_hits=int((target&masks[1-j]).sum()),edge_breaks=breaks(masks[j],same,sk),center_breaks=breaks(masks[1-j],same,sk)))
        bd.save_table(OUT/'native.csv',detail);bd.save_table(OUT/'seams.csv',seams)
        out=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,threads=torch.get_num_threads(),images=len(pairs),native=totals,selected=selected,seams={})
        for halo in (0,32):
            rows=[r for r in seams if r['halo']==halo]
            out['seams'][halo]={k:sum(r[k] for r in rows) for k in ('points','edge_hits','center_hits','edge_breaks','center_breaks')}
            out['seams'][halo]['p95']=float(np.percentile([r['ms'] for r in rows],95))
        (OUT/'summary.json').write_text(json.dumps(out,indent=2));print(out['seams'])
if __name__=='__main__':main()
