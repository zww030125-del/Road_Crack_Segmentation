"""Compare local BCE weights at the starting model's validation false-positive budget."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from torch.nn import functional as F
from torch.utils.data import DataLoader

import train_full_image_baseline as b
from train_detail_branch_experiment import INIT, prepare_validation, evaluate_gaps
from train_pooling_stride_experiment import make_model
from datasets.splits import get_train_val_pairs

SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out = b.ROOT / 'outputs/local_bce_budget/seed_42'
control = b.ROOT / 'outputs/pooling_stride_experiment/seed_42/stride_2'


def top_indices(score, region, count):
    indices = np.flatnonzero(region)
    if len(indices) <= count:
        return indices
    return indices[np.argsort(-score.ravel()[indices], kind='stable')[:count]]


@torch.inference_mode()
def predict(model, image):
    return model.eval()(image[None].to(device)).sigmoid()[0, 0].cpu().numpy()


class Indexed(b.FullImages):
    def __getitem__(self, i):
        return *super().__getitem__(i), self.pairs[i][0].name


def evaluate(label, model, samples, geometry, budget, original_gap, threshold=None):
    scores = [predict(model, s[0]) for s in samples]
    if threshold is None:
        bg = np.concatenate([p[~s[1]] for p,s in zip(scores,samples)])
        index = len(bg)-budget-1
        threshold = float(np.partition(bg,index)[index])
    tp = fp = fn = low = 0
    for p, (_,truth,thinlow) in zip(scores,samples):
        mask = p > threshold
        tp += int((mask & truth).sum())
        fp += int((mask & ~truth).sum())
        fn += int((~mask & truth).sum())
        low += int((thinlow & ndi.maximum_filter(mask,size=5)).sum())
    point = dict(threshold=threshold,thin_low_covered=low)
    gaps, per_image = evaluate_gaps(model,samples,geometry,point,device)
    b.save_table(out / (label+'_per_image.csv'),per_image)
    idx,gap = original_gap
    near = ndi.maximum_filter(scores[idx],size=5)
    return dict(model=label,threshold=threshold,tp=tp,fp=fp,fn=fn,dice=2*tp/(2*tp+fp+fn),
                thin=gaps['thin_covered'],thin_total=gaps['thin_points'],low=low,low_total=gaps['thin_low_points'],
                long_thin=gaps['long_thin_missing_points'],fixed80=int((near[gap]>threshold).sum()),
                fixed80_median_probability=float(np.median(scores[idx][gap])))


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    out.mkdir(parents=True,exist_ok=False)
    state=torch.load(INIT,map_location='cpu',weights_only=True)
    net=make_model(2,state).to(device)
    train,val=get_train_val_pairs(b.ROOT/'data',seed=42)
    samples,geometry=prepare_validation(val)
    diag=json.loads((INIT.parent/'gap_diagnostics/summary.json').read_text())
    case=diag['gallery'][0]
    idx=next(i for i,g in enumerate(geometry) if g[0]==case['image'])
    t=diag['operating_threshold']
    pred=predict(net,samples[idx][0])>t
    labels,_=ndi.label(geometry[idx][1]&~ndi.maximum_filter(pred,size=5),structure=np.ones((3,3)))
    gap=labels==case['component']
    assert gap.sum()==80 and int((gap & geometry[idx][3]).sum())==34
    results=[evaluate('initial',net,samples,geometry,None,(idx,gap),t)]
    budget=results[0]['fp']
    old=json.loads((INIT.parent/'summary.json').read_text())['precision_target_point']
    assert t==old['threshold'] and budget==old['fp']
    print('FP budget',budget,flush=True)
    frozen_hash=b.encoder_hash(net)
    masks={}
    for (image,truth),pair in zip(b.FullImages(train),train):
        truth=truth[0].numpy()
        thin=skeletonize(truth)&(2*ndi.distance_transform_edt(truth)<=3)
        positive=truth & ndi.maximum_filter(thin,size=3)
        dist=ndi.distance_transform_edt(~truth)
        score=predict(net,image)
        negative=np.zeros_like(truth)
        for region in ((dist>=3)&(dist<=10),dist>10):
            negative.flat[top_indices(score,region, max(1,int(positive.sum())//2))]=True
        masks[pair[0].name]=(positive,negative)
    _,hard=b.rank_training_images(b.FullImages(train))
    pairs=train+[train[i] for i in hard]
    with (control/'training_history.csv').open() as f:
        history=list(csv.DictReader(f))
    ref_manifest=json.loads((control.parent/'summary.json').read_text())
    assert ref_manifest['checkpoint_sha256']==hashlib.sha256(INIT.read_bytes()).hexdigest()
    net.load_state_dict(torch.load(control/'final_model.pth',map_location='cpu',weights_only=True))
    results.append(evaluate('control',net,samples,geometry,budget,(idx,gap)))
    for name in ('positive','both'):
        torch.manual_seed(SEED)
        net=make_model(2,state).to(device)
        opt=torch.optim.Adam((p for p in net.parameters() if p.requires_grad),lr=1e-4)
        loader=DataLoader(Indexed(pairs,augment_rng=np.random.default_rng(SEED)),batch_size=1,shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
        logs=[]
        pw=torch.tensor([b.POS_WEIGHT],device=device)
        for epoch in range(10):
            net.train()
            opt.zero_grad(set_to_none=True)
            audit=hashlib.sha256()
            for step,(image,truth,filename) in enumerate(loader,1):
                audit.update(memoryview(image.numpy()))
                audit.update(memoryview(truth.numpy()))
                pos,neg=masks[filename[0]]
                weight=1+pos.astype(np.float32)
                if name=='both': weight+=neg
                w=torch.from_numpy(weight).to(device)
                logits=net(image.to(device))
                loss=b.objective(logits,truth.to(device,dtype=logits.dtype),
                    lambda a,y:F.binary_cross_entropy_with_logits(a,y,pos_weight=pw,weight=w))
                (loss/b.ACCUMULATION).backward()
                if step % b.ACCUMULATION==0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            assert len(loader)%b.ACCUMULATION==0 and audit.hexdigest()==history[epoch]['training_batches_sha256']
            logs.append(dict(epoch=epoch+1,input_sha256=audit.hexdigest()))
            print(name,epoch+1,flush=True)
        assert b.encoder_hash(net)==frozen_hash
        torch.save(net.state_dict(),out/(name+'.pth'))
        b.save_table(out/(name+'_audit.csv'),logs)
        results.append(evaluate(name,net,samples,geometry,budget,(idx,gap)))
    assert all(r['fp']<=budget for r in results)
    b.save_table(out/'comparison.csv',results)
    print(results,flush=True)

if __name__=='__main__':
    main()
