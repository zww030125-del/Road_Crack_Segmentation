"""Diagnostic validation curves at exactly the reference FP count; thresholds use validation labels."""
import json
import numpy as np
import torch
from scipy import ndimage as nd

from validate_auto_lines import (D, DEST, FullImages, ROOT, bd, counts, geometry,
                                 get_train_val_pairs, line_response, ordered,
                                 original_gaps, save, summarize)

@torch.inference_mode()
def main():
    rate = json.loads((DEST/'lock.json').read_text())['chosen'][0]
    models = bd.f.load()
    _,val = get_train_val_pairs(ROOT/'data',seed=42)
    records = []
    budget = 0
    for x,m in FullImages(val):
        y=m[0].numpy()
        f=bd.f.fields(x[None].to(D),models)
        base=bd.correct(bd.masks(f,models[0][3]),(.085,.065))
        thick=nd.maximum_filter(nd.distance_transform_edt(base),7)>=3.
        edge=base & ~nd.binary_erosion(base,structure=np.ones((3,3))) & thick
        drop=ordered(f[2]+.2*f[1]+.15*f[0],edge,int(base.sum()*rate))
        kept=base.copy()
        kept.flat[drop]=False
        budget+=int((~y.flat[drop]).sum())
        records.append((y,geometry(x,y),base,kept,f[0],line_response(f[0])[0]))
    report={}
    for name,idx in [('raw',4),('line',5)]:
        scores=np.concatenate([r[idx][~r[0]&~r[2]] for r in records])
        j=len(scores)-budget-1
        threshold=float(np.partition(scores,j)[j])
        rows=[]
        for r in records:
            y,g,base,kept=r[:4]
            mask=kept | (~base & (r[idx]>threshold))
            labels,ids=original_gaps(base,g)
            near=nd.maximum_filter(mask,5)
            row=counts(mask,.5,y,g)
            row['original_long_hits']=int((near & np.isin(labels,ids) & g[1]).sum())
            rows.append(row)
        report[name]=dict(threshold=threshold,**summarize(rows))
    save('equal_fp_diagnostic',report)
    print(json.dumps(report))

if __name__=='__main__': main()
