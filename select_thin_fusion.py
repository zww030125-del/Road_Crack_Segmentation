"""Select a width-aware fusion on validation; open test labels after locking it."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.special import expit
from PIL import Image
import end_to_end_thin as e
from thin_followup import geometry,counts,summarize
from train_full_image_baseline import ROOT,FullImages,save_table
from datasets.splits import get_train_val_pairs

OUT=e.OUT/'width_fusion'

def load():
    old=e.previous()
    models=[]
    for seed in (42,43):
        net=e.Specialist().to(e.DEVICE)
        net.load_state_dict(torch.load(e.OUT/f'specialist_{seed}.pth',map_location=e.DEVICE,weights_only=True))
        models.append(net.eval())
    return old,models


def representation(x,models,old):
    original,seg,center=e.fields(x,models,old)
    # Local foreground occupancy estimates the width of well supported structures.
    support=ndi.uniform_filter(expit(original-old[3]),size=15,mode='nearest')
    return original,seg,center,support


def fuse(data,spec):
    original,seg,center,support=data
    a,c,w=spec
    return (1-a)*original+a*seg+c*center-w*support


def measure(pairs,models,old,configs,split):
    obs=[]
    for (im,mask),pair in zip(FullImages(pairs),pairs):
        y=mask[0].numpy()
        fields=representation(im[None].to(e.DEVICE),models,old)
        obs.append((pair[0].name,y,geometry(im,y),fields))
    metrics=[]
    for spec,threshold in configs:
        maps=[fuse(r[3],spec) for r in obs]
        if threshold is None:
            bg=np.concatenate([z[~r[1]] for z,r in zip(maps,obs)])
            idx=len(bg)-49000-1;threshold=float(np.partition(bg,idx)[idx]);del bg
        rows=[]
        for (name,y,g,fs),z in zip(obs,maps):
            before=ndi.maximum_filter(fs[0],5)>old[3]
            after=ndi.maximum_filter(z,5)>threshold
            rows.append(dict(image=name,**counts(z,threshold,y,g),gained=int((~before&after&g[1]).sum()),lost=int((before&~after&g[1]).sum())))
        result=dict(spec=spec,threshold=threshold,**summarize(rows));metrics.append(result)
        if split=='test':save_table(OUT/('test_'+'_'.join(map(str,spec))+'.csv'),rows)
    return metrics


def main():
    OUT.mkdir(exist_ok=False)
    old,models=load()
    _,val=get_train_val_pairs(ROOT/'data',seed=42)
    first=json.loads((e.OUT/'selection.json').read_text())['selected']
    a,b=first['mix'];initial=(a,a*b,0.)
    configs=[((0.,0.,0.),old[3])]
    if a:configs.append((initial,first['threshold']))
    configs.extend(((alpha,cl,width),None) for alpha in (.25,.5,.75) for cl in (0.,.35) for width in (2.,5.,9.))
    rows=measure(val,models,old,configs,'val');ref=rows[0]
    valid=[r for r in rows[1:] if r['thin']>ref['thin'] and r['low']>ref['low'] and r['long']<ref['long'] and r['dice']>=ref['dice']-.003 and r['fp']<=ref['fp']]
    best=max(valid,key=lambda r:r['thin_recall']+r['low_recall']) if valid else ref
    (OUT/'locked.json').write_text(json.dumps(dict(selected=best,validation=rows),indent=2))
    print('validation locked:',best,flush=True)
    if best==ref:return
    pairs=[(p,ROOT/'data/test_lab'/(p.stem+'.png')) for p in sorted((ROOT/'data/test_img').glob('*.jpg'))]
    assert len(pairs)==237
    res=measure(pairs,models,old,[(r['spec'],r['threshold']) for r in (ref,best)],'test')
    (OUT/'test.json').write_text(json.dumps(res,indent=2))
    print('test',res,flush=True)


def infer(paths):
    source,destination=paths
    if destination.exists():raise FileExistsError(destination)
    old,models=load()
    config=json.loads((OUT/'locked.json').read_text())['selected']
    with Image.open(source) as im:
        x=torch.from_numpy(np.array(im.convert('RGB'),dtype='float32')/255).permute(2,0,1).contiguous()[None].to(e.DEVICE)
    z=fuse(representation(x,models,old),config['spec'])
    Image.fromarray((z>config['threshold']).astype('uint8')*255).save(destination,format='PNG')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--infer',type=Path,nargs=2)
    args=p.parse_args()
    if args.infer:infer(args.infer)
    else:main()
