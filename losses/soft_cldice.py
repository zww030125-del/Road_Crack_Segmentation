
import torch
import torch.nn.functional as F


def soft_erode(x):
    """通过最小池化近似腐蚀"""
    vertical = -F.max_pool2d(
        -x, (3, 1), stride=1, padding=(1, 0)
    )
    horizontal = -F.max_pool2d(
        -x, (1, 3), stride=1, padding=(0, 1)
    )
    return torch.minimum(vertical, horizontal)


def soft_dilate(x):
    """通过最大池化近似膨胀"""
    return F.max_pool2d(
        x, 3, stride=1, padding=1
    )


def soft_open(x):
    return soft_dilate(soft_erode(x))


def soft_skeleton(x, iterations=10):
    """
    可微分的近似骨架提取。
    x: [B, 1, H, W]，范围0~1
    """
    opened = soft_open(x)
    skel = F.relu(x - opened)

    for _ in range(iterations):
        x = soft_erode(x)
        opened = soft_open(x)

        delta = F.relu(x - opened)
        skel = skel + F.relu(
            delta - skel * delta
        )

    return skel


def soft_cldice_loss(logits, targets, iterations=10):
    """
    logits: 网络直接输出
    targets: 二值GT，形状与logits相同

    返回可参与反向传播的soft-clDice损失。
    """

    probs = torch.sigmoid(logits)

    pred_skel = soft_skeleton(
        probs, iterations
    )
    gt_skel = soft_skeleton(
        targets, iterations
    )

    dims = (1, 2, 3)
    smooth = 1.0

    tprec = (
        (pred_skel * targets).sum(dim=dims)
        + smooth
    ) / (
        pred_skel.sum(dim=dims) + smooth
    )

    tsens = (
        (gt_skel * probs).sum(dim=dims)
        + smooth
    ) / (
        gt_skel.sum(dim=dims) + smooth
    )

    cldice = (
        2 * tprec * tsens
        / (tprec + tsens).clamp_min(1e-6)
    )

    # 对完全没有真实裂纹的裁剪块，
    # 保留BCE和Dice监督，不计入骨架损失。
    valid = targets.sum(dim=dims) > 0

    if valid.any():
        return 1 - cldice[valid].mean()

    return logits.sum() * 0.0
