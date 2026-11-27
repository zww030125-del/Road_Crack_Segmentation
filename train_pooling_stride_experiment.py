"""Paired frozen-encoder continuation: maxpool stride 2 versus stride 1."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import train_full_image_baseline as baseline
from datasets.splits import get_train_val_pairs
from models.resnet_unet import ResNetUNet
from train_detail_branch_experiment import (
    INIT, AuditedBatches, prepare_validation, evaluate_gaps,
)


def make_model(stride, state, early_finetune=False):
    model = ResNetUNet(pretrained=False, freeze_encoder=True)
    model.load_state_dict(state, strict=True)
    model.encoder.maxpool.stride = stride
    if early_finetune:
        # Keep all BN affine parameters and running statistics fixed, even in train mode.
        for block in (model.encoder.conv1, model.encoder.layer1):
            for module in block.modules():
                if isinstance(module, nn.Conv2d):
                    for parameter in module.parameters():
                        parameter.requires_grad = True
    return model


def diagnose(model, validation, geometry, device, output, label):
    old = baseline.THRESHOLDS
    try:
        baseline.THRESHOLDS = np.arange(5, 996) / 1000
        curve = baseline.validate(model, validation, device)
    finally:
        baseline.THRESHOLDS = old
    baseline.save_table(output / f'{label}_thresholds.csv', curve)
    eligible = [r for r in curve if r['precision'] >= baseline.PRECISION_TARGET]
    point = max(eligible, key=lambda r: (r['recall'], r['dice']), default=None)
    gaps = None
    if point is not None:
        gaps, rows = evaluate_gaps(model, validation, geometry, point, device)
        baseline.save_table(output / f'{label}_per_image.csv', rows)
    return {'best_grid': max(curve, key=lambda r: r['dice']),
            'precision_target_point': point, 'gap_metrics': gaps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--early-finetune', action='store_true',
                        help='Train conv1/layer1 convolution weights at 1e-5; keep all BN frozen')
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        parser.error('Invalid epochs or seed')
    experiment = 'pooling_early_finetune' if args.early_finetune else 'pooling_stride_experiment'
    output = args.output or baseline.ROOT / 'outputs' / experiment / f'seed_{args.seed}'
    if output.exists():
        parser.error(f'Output exists: {output}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    state = torch.load(INIT, map_location='cpu', weights_only=True)
    train, val = get_train_val_pairs(baseline.ROOT / 'data', seed=baseline.SPLIT_SEED)
    _, hard = baseline.rank_training_images(baseline.FullImages(train))
    pairs = train + [train[i] for i in hard]
    validation, geometry = prepare_validation(val)
    output.mkdir(parents=True, exist_ok=False)
    result = {'seed': args.seed, 'epochs': args.epochs, 'split_seed': baseline.SPLIT_SEED,
              'train_images': len(train), 'validation_images': len(val),
              'draws_per_epoch': len(pairs), 'checkpoint': str(INIT),
              'checkpoint_sha256': hashlib.sha256(INIT.read_bytes()).hexdigest(),
              'learning_rate': 1e-4, 'primary_protocol': 'fixed final epoch',
              'early_finetune': args.early_finetune,
              'encoder_learning_rate': 1e-5 if args.early_finetune else None,
              'device': str(device), 'torch_version': str(torch.__version__), 'variants': []}
    reference_hashes = None
    for stride in (2, 1):
        folder = output / f'stride_{stride}'
        folder.mkdir()
        torch.manual_seed(args.seed)
        model = make_model(stride, state, args.early_finetune).to(device)
        trainable_encoder = {n for n, p in model.encoder.named_parameters() if p.requires_grad}
        before = baseline.encoder_hash(model)
        initial = diagnose(model, validation, geometry, device, folder, 'initial')
        loader = AuditedBatches(DataLoader(
            baseline.FullImages(pairs, augment_rng=np.random.default_rng(args.seed)),
            batch_size=1, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(args.seed)))
        groups = [{'params': [p for n, p in model.named_parameters()
                              if p.requires_grad and not n.startswith('encoder.')], 'lr': 1e-4}]
        if args.early_finetune:
            groups.append({'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': 1e-5})
        optimizer = torch.optim.Adam(groups)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([baseline.POS_WEIGHT], device=device))
        history = []
        best = None
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            loss = baseline.train_epoch(model, loader, optimizer, bce, device)
            curve = baseline.validate(model, validation, device)
            point = max(curve, key=lambda r: r['dice'])
            row = {'epoch': epoch, 'loss': loss, **point,
                   'training_batches_sha256': loader.digest.hexdigest(),
                   'elapsed_seconds': time.perf_counter() - start}
            history.append(row)
            baseline.save_table(folder / 'training_history.csv', history)
            if best is None or point['dice'] > best['dice']:
                best = row
                torch.save(model.state_dict(), folder / 'best_model.pth')
            print(f'stride={stride} epoch={epoch}/{args.epochs} loss={loss:.5f} Dice={point["dice"]:.6f}', flush=True)
        changed = []
        for name, tensor in model.encoder.state_dict().items():
            if not torch.equal(tensor.cpu(), state['encoder.' + name]):
                if name not in trainable_encoder:
                    raise RuntimeError(f'Frozen encoder tensor changed: {name}')
                changed.append(name)
        if args.early_finetune and set(changed) != trainable_encoder:
            raise RuntimeError('Expected early convolution weights did not all update')
        if not args.early_finetune and baseline.encoder_hash(model) != before:
            raise RuntimeError('Frozen encoder changed')
        torch.save(model.state_dict(), folder / 'final_model.pth')
        final = diagnose(model, validation, geometry, device, folder, 'final')
        model.load_state_dict(torch.load(folder / 'best_model.pth', map_location='cpu', weights_only=True))
        selected = diagnose(model, validation, geometry, device, folder, 'best')
        hashes = [r['training_batches_sha256'] for r in history]
        if reference_hashes is not None:
            assert hashes == reference_hashes, 'Different augmented training batches'
        reference_hashes = hashes
        summary = {'stride': stride, 'best_epoch': best['epoch'], 'encoder_sha256': before,
                   'trainable_encoder_names': sorted(trainable_encoder),
                   'changed_encoder_names': sorted(changed), 'frozen_encoder_tensors_unchanged': True,
                   'initial': initial, 'final': final, 'best': selected,
                   'params': sum(p.numel() for p in model.parameters()),
                   'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        result['variants'].append(summary)
        (output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        del model, optimizer
    result['input_hashes_match_each_epoch'] = True
    (output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
