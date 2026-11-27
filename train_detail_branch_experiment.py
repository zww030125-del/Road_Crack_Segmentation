"""Compare native- and half-resolution RGB detail branches on the full-image baseline."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.splits import get_train_val_pairs
from models.resnet_unet import ResNetUNet
from train_full_image_baseline import (
    ROOT, SPLIT_SEED, POS_WEIGHT, PRECISION_TARGET, THIN_WIDTH, FullImages,
    encoder_hash, rank_training_images, save_table, thin_low_skeleton,
    train_epoch, validate,
)


VARIANTS = ("base", "half", "full")
INIT = ROOT / "outputs" / "full_image_baseline" / "seed_42" / "best_model.pth"
LEARNING_RATE = 1e-4
LONG_GAP = 20
CONNECTIVITY = np.ones((3, 3), dtype=bool)


class DetailResNetUNet(ResNetUNet):
    def __init__(self, half_resolution):
        super().__init__(pretrained=False, freeze_encoder=True)
        self.half_resolution = half_resolution
        self.detail = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=1),
        )
        nn.init.zeros_(self.detail[-1].weight)
        nn.init.zeros_(self.detail[-1].bias)

    def forward(self, rgb):
        x = (rgb - self.mean) / self.std
        encoder = self.encoder
        e0 = encoder.relu(encoder.bn1(encoder.conv1(x)))
        e1 = encoder.layer1(encoder.maxpool(e0))
        e2 = encoder.layer2(e1)
        e3 = encoder.layer3(e2)
        e4 = encoder.layer4(e3)
        d4 = self.dec4(self.up_and_concat(e4, e3))
        d3 = self.dec3(self.up_and_concat(d4, e2))
        d2 = self.dec2(self.up_and_concat(d3, e1))
        d1 = self.dec1(self.up_and_concat(d2, e0))

        size = rgb.shape[-2:]
        detail_rgb = rgb
        if self.half_resolution:
            detail_rgb = F.avg_pool2d(rgb, kernel_size=2, stride=2, ceil_mode=True,
                                      count_include_pad=False)
        detail = self.detail(detail_rgb)
        if self.half_resolution:
            detail = F.interpolate(detail, size=size, mode="bilinear", align_corners=False)
        decoder = F.interpolate(d1, size=size, mode="bilinear", align_corners=False)
        return self.output(self.dec0(torch.cat([decoder + detail, rgb], dim=1)))


def make_model(variant):
    if variant == "base":
        return ResNetUNet(pretrained=False, freeze_encoder=True)
    if variant in ("half", "full"):
        return DetailResNetUNet(half_resolution=variant == "half")
    raise ValueError(f"Unknown variant: {variant}")


def initialize_model(variant, state):
    model = make_model(variant)
    if variant == "base":
        model.load_state_dict(state)
    else:
        loaded = model.load_state_dict(state, strict=False)
        allowed = {f"detail.{name}" for name in model.detail.state_dict()}
        if set(loaded.missing_keys) != allowed or loaded.unexpected_keys:
            raise ValueError(f"Incompatible starting checkpoint: {loaded}")
    return model


class AuditedBatches:
    """Hash the actual augmented tensors and order passed to the training loop."""

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.digest = hashlib.sha256()
        for images, masks in self.loader:
            self.digest.update(memoryview(images.numpy()))
            self.digest.update(memoryview(masks.numpy()))
            yield images, masks


def prepare_validation(pairs):
    validation = []
    geometry = []
    for pair, (image, mask) in zip(pairs, FullImages(pairs)):
        truth = mask.numpy()[0]
        skeleton = skeletonize(truth)
        thin = skeleton & (2 * ndi.distance_transform_edt(truth) <= THIN_WIDTH)
        thin_low, _ = thin_low_skeleton(image.numpy(), truth)
        validation.append((image, truth, thin_low))
        geometry.append((pair[0].name, skeleton, thin, thin_low))
    return validation, geometry


@torch.inference_mode()
def evaluate_gaps(model, samples, geometry, point, device):
    model.eval()
    per_image = []
    for (image, _, _), (name, skeleton, thin, thin_low) in zip(samples, geometry):
        score = model(image[None].to(device)).sigmoid()[0, 0].cpu().numpy()
        prediction = score.astype(np.float64) > point["threshold"]
        covered = ndi.binary_dilation(prediction, structure=CONNECTIVITY, iterations=2)
        labels, components = ndi.label(skeleton & ~covered, structure=CONNECTIVITY)
        sizes = np.bincount(labels.ravel(), minlength=components + 1)
        thin_counts = np.bincount(labels[thin], minlength=components + 1)
        thin_low_counts = np.bincount(labels[thin_low], minlength=components + 1)
        long = sizes >= LONG_GAP
        long[0] = False
        per_image.append({
            "image": name,
            "thin_points": int(thin.sum()),
            "thin_covered": int(thin_counts[0]),
            "thin_low_points": int(thin_low.sum()),
            "thin_low_covered": int(thin_low_counts[0]),
            "long_thin_missing_points": int(thin_counts[long].sum()),
            "long_thin_low_missing_points": int(thin_low_counts[long].sum()),
            "long_components_touching_thin": int(np.count_nonzero(long & (thin_counts > 0))),
            "longest_missing_component_touching_thin": int(
                sizes[1:][thin_counts[1:] > 0].max(initial=0)
            ),
        })
    aggregate = {
        key: sum(row[key] for row in per_image) for key in per_image[0]
        if key not in ("image", "longest_missing_component_touching_thin")
    }
    key = "longest_missing_component_touching_thin"
    longest = max(per_image, key=lambda row: row[key])
    aggregate[key] = longest[key]
    aggregate["longest_case"] = longest["image"]
    aggregate["thin_recall_2px"] = aggregate["thin_covered"] / aggregate["thin_points"]
    if aggregate["thin_low_covered"] != point["thin_low_covered"]:
        raise RuntimeError("Gap analysis and threshold sweep disagree on thin-low coverage")
    return aggregate, per_image


def fit_variant(variant, state, initial_prediction, epoch_pairs, validation, geometry,
                args, output, device):
    torch.manual_seed(args.seed)
    model = initialize_model(variant, state).to(device)
    encoder_before = encoder_hash(model)
    model.eval()
    with torch.inference_mode():
        actual = model(validation[0][0][None].to(device)).cpu()
    if not torch.equal(actual, initial_prediction):
        raise RuntimeError(f"{variant} changed the initial prediction")
    loader = AuditedBatches(DataLoader(
        FullImages(epoch_pairs, augment_rng=np.random.default_rng(args.seed)),
        batch_size=1, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    ))
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device))
    output.mkdir()
    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, bce, device)
        curve = validate(model, validation, device)
        point = max(curve, key=lambda row: row["dice"])
        eligible = [row for row in curve if row["precision"] >= PRECISION_TARGET]
        matched = max(eligible, key=lambda row: (row["recall"], row["dice"]), default=None)
        row = {"epoch": epoch, "loss": loss, **point,
               "training_batches_sha256": loader.digest.hexdigest()}
        history.append(row)
        save_table(output / "training_history.csv", history)
        print(f"{variant} {epoch:02d}/{args.epochs} | loss={loss:.4f} | "
              f"best-grid Dice={point['dice']:.5f} | "
              f"P-target={matched and round(matched['precision'], 5)}", flush=True)
        if best is None or row["dice"] > best["dice"]:
            best, best_matched = row, matched
            torch.save(model.state_dict(), output / "best_model.pth")
            save_table(output / "validation_thresholds.csv", curve)
    if encoder_hash(model) != encoder_before:
        raise RuntimeError(f"{variant} changed the frozen encoder")
    model.load_state_dict(torch.load(output / "best_model.pth", map_location="cpu", weights_only=True))
    gap_metrics = None
    if best_matched is not None:
        gap_metrics, per_image = evaluate_gaps(model, validation, geometry, best_matched, device)
        save_table(output / "per_image_gaps.csv", per_image)
    summary = {
        "variant": variant, "best": best, "precision_target_point": best_matched,
        "gap_metrics": gap_metrics, "encoder_sha256": encoder_before,
        "params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary, [row["training_batches_sha256"] for row in history]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Training seed, with fixed data split")
    parser.add_argument("--epochs", type=int, default=10, help="Continuation epochs per variant")
    parser.add_argument("--output", type=Path, help="New run directory")
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        parser.error("--epochs must be positive and --seed nonnegative")
    output = args.output or ROOT / "outputs" / "detail_branch_experiment" / f"seed_{args.seed}"
    if output.exists():
        parser.error(f"Output already exists: {output}. Specify a new --output.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    state = torch.load(INIT, map_location="cpu", weights_only=True)
    train, val = get_train_val_pairs(ROOT / "data", seed=SPLIT_SEED)
    if not train or not val:
        raise ValueError("Both training and validation splits must contain images")
    _, hard_indices = rank_training_images(FullImages(train))
    epoch_pairs = train + [train[i] for i in hard_indices]
    validation, geometry = prepare_validation(val)
    model = initialize_model("base", state).to(device).eval()
    with torch.inference_mode():
        initial_prediction = model(validation[0][0][None].to(device)).cpu()
    del model
    output.mkdir(parents=True, exist_ok=False)
    print(f"{device} | seed={args.seed} | {len(train)} train/{len(val)} val | "
          f"{len(epoch_pairs)} training draws/epoch", flush=True)
    summaries = []
    reference_hashes = None
    for variant in VARIANTS:
        summary, hashes = fit_variant(
            variant, state, initial_prediction, epoch_pairs, validation, geometry,
            args, output / variant, device,
        )
        if reference_hashes is not None and hashes != reference_hashes:
            raise RuntimeError(f"{variant} received different training inputs")
        reference_hashes = hashes
        summaries.append(summary)
    result = {
        "seed": args.seed, "split_seed": SPLIT_SEED, "epochs_per_variant": args.epochs,
        "train_images": len(train), "validation_images": len(val),
        "draws_per_epoch": len(epoch_pairs), "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "checkpoint": str(INIT), "checkpoint_sha256": hashlib.sha256(INIT.read_bytes()).hexdigest(),
        "precision_target": PRECISION_TARGET, "long_gap_min_points": LONG_GAP,
        "input_hashes_match_each_epoch": True, "variants": summaries,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Completed paired experiment: {output}", flush=True)


if __name__ == "__main__":
    main()
