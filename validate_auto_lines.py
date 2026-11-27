"""Full-frame oriented ridge recovery, with train-locked edits and a same-budget control."""
import csv
import json
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from scipy import ndimage as nd

from paired_context import D, OUT, ROOT, FullImages, bd, get_train_val_pairs
from thin_followup import counts, geometry, summarize

DEST = OUT / 'automatic_lines'
RATES = (.005, .015, .03)
FRACTIONS = (.25, .5, 1.)
CONFIGS = [(r,f) for r in RATES for f in FRACTIONS]

def shifted(padded, shape, dy, dx):
    h,w = shape
    return padded[8+dy:8+dy+h,8+dx:8+dx+w]

def line_response(z):
    pad = np.pad(z,8,mode='reflect')
    best = np.full_like(z,-np.inf)
    winner = np.zeros(z.shape,np.uint8)
    for direction,angle in enumerate(np.arange(8)*np.pi/8):
        tangent = np.array([np.sin(angle),np.cos(angle)])
        dy,dx = np.rint(6*tangent[::-1]*[1,-1]).astype(int)
        ridge = z-np.maximum(shifted(pad,z.shape,dy,dx),shifted(pad,z.shape,-dy,-dx))
        for scale,radius in enumerate((4,8)):
            points = np.unique(np.rint(np.arange(-radius,radius+1)[:,None]*tangent).astype(int),axis=0)
            kernel = np.zeros((2*radius+1,2*radius+1),np.float32)
            kernel[tuple((points+radius).T)] = 1/len(points)
            along = nd.correlate(ridge,kernel,mode="mirror")
            # Both persistent evidence along the line and support at its center matter.
            response = along + .25*ridge + .25*z
            better = response>best
            winner[better] = 2*direction+scale
            np.maximum(best,response,out=best)
    return best,winner

def ordered(score,mask,n,descending=False):
    ids = np.flatnonzero(mask)
    if not len(ids) or n<=0: return ids[:0]
    v = score.ravel()[ids]
    if descending: v = -v
    return ids[np.lexsort((ids,v))[:n]]

def edited(base,drop,add,config):
    rate,fraction = config
    n = min(int(base.sum()*rate),len(drop))
    result = base.copy()
    result.flat[drop[:n]]=False
    result.flat[add[:int(n*fraction)]]=True
    return result

def original_gaps(base,g):
    sk,thin,_,_ = g
    labels,n = nd.label(sk & ~nd.maximum_filter(base,5),np.ones((3,3)))
    size = np.bincount(labels.ravel(),minlength=n+1)
    include = np.unique(labels[thin & (labels>0)])
    ids = include[size[include]>=20]
    return labels,ids

def matched_comparison(z,line,winner,base,y,g,rng):
    pos = np.flatnonzero(g[1] & ~nd.maximum_filter(base,5))
    bg = (nd.distance_transform_edt(~y)>=4)&~base
    records=[]
    for group in range(16):
        pp = pos[winner.flat[pos]==group]
        if not len(pp): continue
        bb = np.flatnonzero(bg & (winner==group))
        if not len(bb): continue
        bb = rng.permutation(bb)
        bb = bb[np.argsort(z.flat[bb],kind='stable')]
        idx = np.searchsorted(z.flat[bb],z.flat[pp])
        left,right = bb[np.maximum(idx-1,0)],bb[np.minimum(idx,len(bb)-1)]
        chosen = np.where(abs(z.flat[left]-z.flat[pp])<abs(z.flat[right]-z.flat[pp]),left,right)
        ok = abs(z.flat[chosen]-z.flat[pp])<=.1
        diff=line.flat[pp[ok]]-line.flat[chosen[ok]]
        records.extend(diff.tolist())
    return records

@torch.inference_mode()
def evaluate(split,pairs,models,configs):
    rows={};differences=[]; blind=[]
    keyrows=list(csv.DictReader((OUT/'key'/'val.csv').open())) if split=='val' else []
    rng=np.random.default_rng(370)
    for pair,(x,m) in zip(pairs,FullImages(pairs)):
        name=pair[0].name
        y=m[0].numpy();g=geometry(x,y)
        fields=bd.f.fields(x[None].to(D),models);z=fields[0]
        base=bd.correct(bd.masks(fields,models[0][3]),(.085,.065))
        line,winner=line_response(z)
        max_n=int(base.sum()*max(RATES))
        # Remove only exterior pixels backed by a broad core, to protect narrow components.
        thick=nd.maximum_filter(nd.distance_transform_edt(base),7)>=3.
        edge=base & ~nd.binary_erosion(base,structure=np.ones((3,3))) & thick
        drop=ordered(fields[2]+.2*fields[1]+.15*z,edge,max_n)
        add={'line':ordered(line,~base,max_n,True),'raw':ordered(z,~base,max_n,True)}
        labels,ids=original_gaps(base,g)
        long_points=g[1]&np.isin(labels,ids)
        if split=='val': differences.extend(matched_comparison(z,line,winner,base,y,g,rng))
        masks={'base':base}
        for config in configs:
            for kind in (('line','raw') if split=='val' else ('line',)):
                masks[f'{kind}_{config}']=edited(base,drop,add[kind],config)
        for tag,mask in masks.items():
            near=nd.maximum_filter(mask,5)
            residual=np.unique(labels[(labels>0)&~near])
            stats=dict(image=name,**counts(mask,.5,y,g),orig_long_n=len(ids),
                       full_long=int(np.count_nonzero(~np.isin(ids,residual))),
                       recovered_long_points=int((near&long_points).sum()),orig_long_points=int(long_points.sum()),
                       add_tp=int((mask&~base&y).sum()),add_fp=int((mask&~base&~y).sum()),
                       del_tp=int((~mask&base&y).sum()),del_fp=int((~mask&base&~y).sum()))
            rows.setdefault(tag,[]).append(stats)
            for case in keyrows:
                if case['image']!=name: continue
                target=np.asarray(Image.open(OUT/'key'/(case['id']+'.png')))>0
                top,left=int(case['top']),int(case['left'])
                blind.append(dict(tag=tag,id=case['id'],image=name,length=case['length'],hits=int(near[top:top+96,left:left+96][target].sum()),n=int(target.sum())))
    if blind: bd.save_table(DEST/'validation_blind.csv',blind)
    if split=='val':
        bd.save_table(DEST/'validation_by_image.csv',[dict(variant=k,**r) for k,group in rows.items() for r in group])
        v=np.array(differences)
        save('background_match',dict(n=len(v),wins=int((v>0).sum()),ties=int((v==0).sum()),median=float(np.median(v))))
    return {k:summarize(v) for k,v in rows.items()}

def save(name,result):
    (DEST/(name+'.json')).write_text(json.dumps(result,indent=2))

def main():
    DEST.mkdir(exist_ok=False)
    models=bd.f.load()
    train,val=get_train_val_pairs(ROOT/'data',seed=42)
    results=evaluate('train',train,models,CONFIGS)
    save('train',results)
    ref=results['base']
    allowed=[c for c in CONFIGS if results[f'line_{c}']['fp']<=ref['fp'] and results[f'line_{c}']['dice']>=ref['dice']]
    candidates=allowed or CONFIGS
    chosen=max(candidates,key=lambda c:(results[f'line_{c}']['recovered_long_points'],results[f'line_{c}']['thin'],results[f'line_{c}']['dice']))
    save('lock',dict(chosen=chosen,eligible=len(allowed)))
    print('locked',chosen,flush=True)
    result=evaluate('val',val,models,[chosen])
    save('validation',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
