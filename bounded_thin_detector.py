"""Bound per-image mask edits using the old mask and the trained foreground specialist."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import torch
from PIL import Image
import fit_joint_thin as f
from datasets.splits import get_train_val_pairs
from train_full_image_baseline import ROOT,FullImages,save_table
from thin_followup import counts,summarize,geometry

OUT=f.OUT/'bounded'
def masks(data,threshold):
    old=data[0]>threshold
    # Low support in the independent, contrast-attenuation trained model identifies suspect borders.
    evidence=data[2]+.2*data[1]
    interior=ndi.binary_erosion(old,structure=np.ones((3,3)),border_value=0)
    drop=np.flatnonzero(old & ~interior & (evidence<3.5))
    add=np.flatnonzero(~old & (evidence>-1.5))
    # A small secondary contribution of the original logit stabilizes orientation and texture.
    score=evidence+.15*np.clip(data[0],-10,10)
    ds=drop[np.argsort(score.ravel()[drop],kind='stable')]
    ad=add[np.argsort(-score.ravel()[add],kind='stable')]
    return old,ds,ad

def correct(cached,config):
    old,drop,add=cached;n=int(old.sum())
    out=old.copy()
    out.flat[drop[:int(config[0]*n)]]=False
    out.flat[add[:int(config[1]*n)]]=True
    return out

def collect(pairs,models):
    records=[]
    for (x,y),pair in zip(FullImages(pairs),pairs):
        y=y[0].numpy()
        fields=f.fields(x[None].to(f.DEVICE),models)
        records.append((pair[0].name,y,geometry(x,y),masks(fields,models[0][3])))
    return records

def evaluate(data,config):
    rows=[]
    for name,y,g,cached in data:
        old=cached[0];pred=correct(cached,config)
        before=ndi.maximum_filter(old,5);after=ndi.maximum_filter(pred,5)
        rows.append(dict(image=name,**counts(pred,.5,y,g),gained=int((~before&after&g[1]).sum()),lost=int((before&~after&g[1]).sum())))
    return rows,dict(config=config,**summarize(rows))

def main():
    OUT.mkdir(exist_ok=False)
    models=f.load()
    _,val=get_train_val_pairs(ROOT/'data',seed=42)
    data=collect(val,models)
    _,ref=evaluate(data,(0.,0.))
    configs=[(drop,add) for drop in (.035,.06,.085,.11,.15) for add in (.02,.04,.065,.09,.12) if add<=drop]
    results=[evaluate(data,c)[1] for c in configs]
    # Margin for transfer; retain the bulk of the older detector's segmentations.
    ok=[r for r in results if r['fp']<=.94*ref['fp'] and r['dice']>=ref['dice']-.003 and
        r['low']>ref['low'] and r['thin']>ref['thin'] and r['long']<ref['long']]
    best=max(ok,key=lambda r:r['thin_recall']+r['low_recall']) if ok else ref
    (OUT/'locked.json').write_text(json.dumps(dict(reference=ref,selected=best,validation=results),indent=2))
    print('locked',best,flush=True)
    if best==ref:return
    del data
    test=[(p,ROOT/'data/test_lab'/(p.stem+'.png')) for p in sorted((ROOT/'data/test_img').glob('*.jpg'))]
    assert len(test)==237
    data=collect(test,models)
    rr=[]
    for tag,c in [('reference',(0.,0.)),('selected',best['config'])]:
        rows,r=evaluate(data,c);rr.append(r);save_table(OUT/f'test_{tag}.csv',rows)
    (OUT/'test.json').write_text(json.dumps(rr,indent=2));print('test',rr,flush=True)

def infer(paths):
    src,dst=paths
    if dst.exists():raise FileExistsError(dst)
    config=json.loads((OUT/'locked.json').read_text())['selected']['config']
    models=f.load()
    with Image.open(src) as im:x=torch.from_numpy(np.array(im.convert('RGB'),dtype='float32')/255).permute(2,0,1).contiguous()[None].to(f.DEVICE)
    pred=correct(masks(f.fields(x,models),models[0][3]),config)
    Image.fromarray(pred.astype('uint8')*255).save(dst,format='PNG')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--infer',type=Path,nargs=2);args=p.parse_args()
    if args.infer:infer(args.infer)
    else:main()
