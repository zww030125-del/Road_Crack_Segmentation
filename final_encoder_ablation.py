"""Paired full-image encoder continuation, selected by validation micro Dice."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from torch.utils.data import DataLoader

import train_full_image_baseline as b
from thin_followup import geometry, counts, summarize
from train_detail_branch_experiment import INIT

OUT=b.ROOT/'outputs'/'final_encoder_ablation'
D=torch.device('cuda')
SEEDS=(42,43,44)
ARMS=('decoder','layer4','full')
EPOCHS=3

def setup(arm):
    net=b.ResNetUNet(pretrained=False,freeze_encoder=True).to(D)
    net.load_state_dict(torch.load(INIT,map_location=D,weights_only=True))
    groups={}
    for name,p in net.named_parameters():
        if not name.startswith('encoder.'):lr=1e-4
        elif name.startswith('encoder.layer4.') and arm!='decoder':lr=3e-5
        elif arm=='full' and not name.startswith('encoder.fc.'):lr=1e-5
        else:continue
        p.requires_grad_(True)
        groups.setdefault(lr,[]).append(p)
    return net,torch.optim.Adam([dict(params=params,lr=lr) for lr,params in groups.items()])

@torch.inference_mode()
def report(net,pairs,threshold):
    net.eval();rows=[]
    for (im,m),pair in zip(b.FullImages(pairs),pairs):
        y=m[0].numpy()
        score=net(im[None].to(D)).sigmoid()[0,0].cpu().numpy()
        rows.append(dict(image=pair[0].name,**counts(score,threshold,y,geometry(im,y))))
    return rows,summarize(rows)

def save(path,data):
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    OUT.mkdir(parents=True,exist_ok=True)
    tr,val=b.get_train_val_pairs(b.ROOT/'data',seed=42)
    test=[(p,b.ROOT/'data/test_lab'/(p.stem+'.png')) for p in sorted((b.ROOT/'data/test_img').glob('*.jpg'))]
    assert (len(tr),len(val),len(test))==(240,60,237)
    samples=[(x,m[0].numpy(),b.thin_low_skeleton(x.numpy(),m[0].numpy())[0]) for x,m in b.FullImages(val)]
    net,_=setup('decoder')
    initial=max(b.validate(net,samples,D),key=lambda v:v['dice'])
    _,start=report(net,val,initial['threshold'])
    save(OUT/'protocol.json',dict(initial_sha256=hashlib.sha256(INIT.read_bytes()).hexdigest(),epochs=EPOCHS,seeds=SEEDS,initial=initial,initial_metrics=start,training='each full image once per epoch; original photometric augmentation and BCE/Dice/soft-clDice; fixed encoder BN statistics'))
    del net
    results=[]
    for seed in SEEDS:
        for arm in ARMS:
            folder=OUT/f'{seed}_{arm}';folder.mkdir(exist_ok=True)
            result=folder/'result.json'
            if result.exists():
                results.append(json.loads(result.read_text()));continue
            torch.manual_seed(seed)
            net,opt=setup(arm)
            loader=DataLoader(b.FullImages(tr,augment_rng=np.random.default_rng(seed)),batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(seed))
            criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([b.POS_WEIGHT],device=D))
            hist=[];best=None
            for epoch in range(1,EPOCHS+1):
                loss=b.train_epoch(net,loader,opt,criterion,D)
                point=max(b.validate(net,samples,D),key=lambda v:v['dice'])
                row=dict(epoch=epoch,loss=loss,**point);hist.append(row)
                if best is None or point['dice']>best['dice']:
                    best=row;torch.save(net.state_dict(),folder/'best.pth')
                print(seed,arm,epoch,round(point['dice'],5),flush=True)
            b.save_table(folder/'history.csv',hist)
            net.load_state_dict(torch.load(folder/'best.pth',map_location=D,weights_only=True))
            rows,metrics=report(net,val,best['threshold'])
            b.save_table(folder/'val.csv',rows)
            r=dict(seed=seed,arm=arm,epoch=best['epoch'],threshold=best['threshold'],**metrics)
            save(result,r);results.append(r)
            del net,opt
    b.save_table(OUT/'validation.csv',results)
    means={a:float(np.mean([r['dice'] for r in results if r['arm']==a])) for a in ARMS}
    winner=max(means,key=means.get)
    save(OUT/'lock.json',dict(means=means,winner=winner))
    tested=[]
    for r in results:
        if r['arm']!=winner:continue
        net,_=setup(winner);folder=OUT/f"{r['seed']}_{winner}"
        net.load_state_dict(torch.load(folder/'best.pth',map_location=D,weights_only=True))
        rows,metrics=report(net,test,r['threshold'])
        b.save_table(folder/'test.csv',rows)
        tested.append(dict(seed=r['seed'],**metrics))
    b.save_table(OUT/'test.csv',tested)
    print('locked',means,winner,flush=True)
if __name__=='__main__':main()
