"""Fixed-threshold paired branch toggles; disabled networks are not executed."""
import tempfile,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as nd
from scipy.special import expit
import field_inference_audit as a

bd=a.bd
OUT=bd.ROOT/'outputs/native_branch_ablation'
ARMS=['bare','detail','contrast','primary','edit_only','specialists','joint','both','legacy']

@torch.inference_mode()
def run():
    OUT.mkdir(exist_ok=False)
    old,special,joint,width=bd.f.load()
    base,heads,key,t=old
    alpha,center,penalty=width['spec']
    assert center==0
    def predict(rgb,arm):
        x=a.tensor(rgb)
        if arm=='bare':return base.net(x)[0,0].cpu().numpy()>t
        if arm=='detail':return base(x)[0,0].cpu().numpy()>t
        if arm=='contrast':return bd.f.e.prior.predict(base,heads,x,['contrast'])['contrast']>t
        if arm=='legacy':return a.decide(bd.f.fields(x,(old,special,joint,width)),(old,special,joint,width),True)
        z=bd.f.e.prior.predict(base,heads,x,[key])[key]
        if arm=='primary':return z>t
        support=nd.uniform_filter(expit(z-t),size=15,mode='nearest')
        seg=0.
        if arm in ('specialists','both'):
            maps=[]
            for net in special:
                m=net(x);n=net(x.flip([-1])).flip([-1])
                maps.append(((m+n)/2)[0].cpu().numpy())
            seg=np.mean(maps,axis=0)[0]
        w=(1-alpha)*z+alpha*seg-penalty*support
        j=0.
        if arm in ('joint','both'):
            for k in range(4):
                v=joint(torch.rot90(x,k,(-2,-1)).contiguous())
                j+=torch.rot90(v,-k,(-2,-1))[0,0].cpu().numpy()/4
        return bd.correct(bd.masks((z,w,j),t),a.CONFIG)
    _,pairs=bd.get_train_val_pairs(bd.ROOT/'data',seed=42)
    rows=[];rng=np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmp:
        dst=Path(tmp)/'pred.png'
        for arm in ARMS:
            for _ in range(3):a.timed(lambda rgb:predict(rgb,arm),pairs[0][0],dst)
        for path,lab in pairs:
            with Image.open(path) as im:x=a.tensor(im)[0].cpu()
            with Image.open(lab) as im:y=np.asarray(im.convert('L'))>127
            g=a.geometry(x,y);verify={}
            for arm in rng.permutation(ARMS):
                m,ms=a.timed(lambda rgb:predict(rgb,arm),path,dst)
                if arm in ('both','legacy'):verify[arm]=m
                rows.append(dict(image=path.name,arm=arm,ms=ms,**a.counts(m,.5,y,g),**a.review(m,y,g)))
            assert np.array_equal(verify['both'],verify['legacy'])
    bd.save_table(OUT/'per_image.csv',rows)
    results={}
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm]
        totals=a.summarize([{k:v for k,v in r.items() if k not in ('arm','ms')} for r in group])
        results[arm]=dict(**totals,p95=float(np.percentile([r['ms'] for r in group],95)))
        print(arm,*[round(results[arm][k],4) for k in ('dice','low_recall','false_regions','long_gaps','fp','p95')])
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':run()
