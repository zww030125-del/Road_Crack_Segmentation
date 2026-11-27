"""Audit blind-traced missed skeletons against the exact pre-edit full-image score."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as nd
from skimage.morphology import skeletonize

from paired_context import D, OUT, ROOT, FullImages, bd, get_train_val_pairs

def read_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))

def sides(point, skeleton, mask, scores, radii):
    y,x = point
    h,w = mask.shape
    top,left = max(0,y-5),max(0,x-5)
    offsets = np.argwhere(skeleton[top:y+6,left:x+6]) + [top-y,left-x]
    offsets = offsets[np.linalg.norm(offsets,axis=1)<=5]
    if len(offsets)<2:
        return None
    values,vectors = np.linalg.eigh(np.cov(offsets.T))
    if values[1]<=1.5*values[0]:
        return None
    normal = vectors[:,0]
    values = []
    for sign in (-1,1):
        pts = np.unique(np.rint(point + sign*radii[:,None]*normal).astype(int),axis=0)
        idx = (pts[:,0]>=0)&(pts[:,0]<h)&(pts[:,1]>=0)&(pts[:,1]<w)
        yy,xx = pts[idx].T
        background = ~mask[yy,xx]
        if background.sum()<2:
            return None
        values.append(float(np.median(scores[yy[background],xx[background]])))
    return values

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces',type=Path)
    args=parser.parse_args()
    traces=json.loads(args.traces.read_text(encoding="utf-8-sig"))
    reviewed={r['id']:r for r in read_rows(OUT/'blind_scores.csv')}
    models=bd.f.load()
    threshold=float(models[0][3])
    rows=[]
    splits=get_train_val_pairs(ROOT/'data',seed=42)
    for split,pairs in zip(('train','val'),splits):
        keys=read_rows(OUT/'key'/(split+'.csv'))
        for pair in pairs:
            cases=[r for r in keys if r['image']==pair[0].name]
            if not cases: continue
            x,m=FullImages([pair])[0]
            mask=m[0].numpy()
            sk=skeletonize(mask)
            fields=bd.f.fields(x[None].to(D),models)
            score=fields[0]
            cached=bd.masks(fields,threshold)
            before=nd.maximum_filter(cached[0],5)
            after=nd.maximum_filter(bd.correct(cached,(.085,.065)),5)
            for case in cases:
                token=case['id']
                target=np.asarray(Image.open(OUT/'key'/(token+'.png')))>0
                line=Image.new('L',(96,96))
                draw=ImageDraw.Draw(line)
                for stroke in traces[token]['strokes']:
                    pts=list(map(tuple,stroke))
                    if len(pts)>1: draw.line(pts,fill=1)
                    elif pts: draw.point(pts[0],fill=1)
                traced=nd.maximum_filter(np.asarray(line),5)>0
                assert np.isclose(traced[target].mean(),float(reviewed[token]['target_coverage']))
                for ly,lx in np.argwhere(target):
                    y,x0=int(case['top'])+ly,int(case['left'])+lx
                    assert sk[y,x0] and not after[y,x0]
                    r=dict(split=split,image=case['image'],id=token,length=case['length'],low=case['low'],y=y,x=x0,traced=int(traced[ly,lx]),precovered=int(before[y,x0]),score=float(score[y,x0]),threshold=threshold)
                    for tag,radii in [('near',np.arange(4,9)),('far',np.arange(8,13))]:
                        v=sides(np.array([y,x0]),sk,mask,score,radii)
                        r[tag+'_left'],r[tag+'_right']=v if v else ('','')
                    rows.append(r)
    dest=OUT/'blind_score_profiles.csv'
    with dest.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    for split in ('train','val'):
        part=[r for r in rows if r['split']==split and r['traced']]
        def stats(sub):
            result=[]
            for tag in ('near','far'):
                good=[r for r in sub if r[tag+'_left']!='']
                diff=[r['score']-max(r[tag+'_left'],r[tag+'_right']) for r in good]
                result.append((len(good),sum(v>0 for v in diff)))
            return len(sub),sum(r['precovered'] for r in sub),round(float(np.median([r['score'] for r in sub])),3),result
        print(split,'threshold',threshold,stats(part))
        for token in sorted({r['id'] for r in part}):
            group=[r for r in part if r['id']==token]
            print(group[0]['image'],group[0]['length'],stats(group))

if __name__=='__main__': main()
