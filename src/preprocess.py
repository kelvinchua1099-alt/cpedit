"""
preprocess.py — image preprocessing utilities for edit-region extraction.

Core function: diff_crop_resize
  1. 逐像素相减找到变化区域
  2. 用最小外接矩形框住变化区域，并以 padding_scale 扩展
  3. 将矩形调整为与原图相同的宽高比（扩展短边，不裁切）
  4. 将矩形平移至图像范围内（保持尺寸不变）
  5. 对 src / tgt 图像各自 crop 后 resize 回原始分辨率
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _clean_mask(mask: np.ndarray, open_iter: int, keep_frac: float) -> np.ndarray:
    """
    Despeckle a raw diff mask so the bbox tracks the real edit region.

    High-contrast renders produce scattered high-diff pixels along
    anti-aliased edges across the *whole* frame; a few outliers stretch the
    global bbox to cover everything. We:
      1. morphological opening — erode away thin edge noise (anti-aliasing),
         keep solid blobs (the actual added/changed object);
      2. keep only connected components whose area is ≥ keep_frac × the
         largest component, dropping isolated specks.
    """
    if open_iter > 0:
        opened = ndimage.binary_opening(mask, iterations=open_iter)
        if opened.any():
            mask = opened   # fall back to raw mask if opening erased everything

    lbl, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    keep_labels = np.where(sizes >= keep_frac * sizes.max())[0] + 1
    return np.isin(lbl, keep_labels)


def _shift_bbox_into_image(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    """
    平移矩形框使其完全落在 [0,W]×[0,H] 内，不改变矩形尺寸。
    若矩形本身大于图像则直接 clamp 到图像边界。
    """
    w = x2 - x1
    h = y2 - y1
    # 若 box 比图像还大，将尺寸 clamp 到图像
    w = min(w, float(W))
    h = min(h, float(H))
    # 先处理左/上越界：整体右移/下移
    if x1 < 0:
        x1, x2 = 0.0, w
    if y1 < 0:
        y1, y2 = 0.0, h
    # 再处理右/下越界：整体左移/上移
    if x2 > W:
        x1, x2 = W - w, float(W)
    if y2 > H:
        y1, y2 = H - h, float(H)
    return int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diff_crop_resize(
    img_src: Image.Image,
    img_tgt: Image.Image,
    padding_scale: float = 1.5,
    diff_threshold: int = 15,
    blur_radius: int = 3,
    open_iter: int = 3,
    keep_frac: float = 0.15,
) -> Tuple[Image.Image, Image.Image, Dict]:
    """
    通过逐像素相减定位两张图片的修改区域，裁剪出该区域并 resize 回原图尺寸。

    流程
    ----
    1. 计算 |img_src - img_tgt| 的逐像素最大差值（对 RGB 三通道取 max）
    2. GaussianBlur 降噪后用 diff_threshold 二值化
    3. 找变化像素的最小外接矩形 bbox_diff
    4. 以 padding_scale 向四周等比扩展
    5. 调整宽高比 = 原图宽高比（扩展短边，中心不变）
    6. 平移使 bbox 完全在图像内（不改变尺寸）
    7. 对 img_src / img_tgt 各自 crop → BICUBIC resize → 原图尺寸

    Args
    ----
    img_src        : 参考图（source / 未编辑）
    img_tgt        : 编辑图（target / 已编辑）；必须与 img_src 同尺寸
    padding_scale  : 在差异框外再扩展的倍率（默认 1.5 即扩展 50%）
    diff_threshold : 像素差绝对值低于此阈值视为"未变化"（0-255）
    blur_radius    : 二值化前的 GaussianBlur 半径（0 = 不模糊）

    Returns
    -------
    crop_src  : img_src 在修改区域的 crop，resize 回原图尺寸
    crop_tgt  : img_tgt 在修改区域的 crop，resize 回原图尺寸
    meta      : 包含以下字段的 dict
        found_change : bool，是否检测到明显差异
        bbox_diff    : (x1,y1,x2,y2) 差异紧凑框（像素坐标），无差异时为 None
        bbox_crop    : (x1,y1,x2,y2) 最终裁剪框
        original_size: (W, H)
    """
    W, H = img_src.size
    if img_tgt.size != (W, H):
        raise ValueError(
            f"img_src and img_tgt must be the same size, "
            f"got {img_src.size} vs {img_tgt.size}"
        )

    # ── 1. 逐像素绝对差（对 RGB 取 max channel） ──────────────────────────
    arr_src = np.array(img_src.convert("RGB"), dtype=np.float32)   # [H,W,3]
    arr_tgt = np.array(img_tgt.convert("RGB"), dtype=np.float32)
    diff    = np.abs(arr_src - arr_tgt).max(axis=-1).astype(np.uint8)  # [H,W]

    # ── 2. 高斯模糊 + 阈值二值化 ─────────────────────────────────────────
    diff_pil = Image.fromarray(diff)
    if blur_radius > 0:
        diff_pil = diff_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask = np.array(diff_pil) > diff_threshold   # [H,W] bool

    # ── 2b. 去散点噪声：形态学开运算 + 最大连通块 ───────────────────────
    if mask.any():
        mask = _clean_mask(mask, open_iter=open_iter, keep_frac=keep_frac)

    # ── 3. 无变化区域：直接返回原图 ──────────────────────────────────────
    ys, xs = np.where(mask)
    if len(xs) == 0:
        meta: Dict = {
            "found_change":  False,
            "bbox_diff":     None,
            "bbox_crop":     (0, 0, W, H),
            "original_size": (W, H),
        }
        return img_src.copy(), img_tgt.copy(), meta

    # ── 4. 差异紧凑框 ────────────────────────────────────────────────────
    x1d, y1d = int(xs.min()), int(ys.min())
    x2d, y2d = int(xs.max()), int(ys.max())
    bbox_diff = (x1d, y1d, x2d, y2d)

    # ── 5. 以 padding_scale 向四周扩展（以差异中心为锚点） ───────────────
    cx = (x1d + x2d) / 2.0
    cy = (y1d + y2d) / 2.0
    w_half = (x2d - x1d) / 2.0 * padding_scale
    h_half = (y2d - y1d) / 2.0 * padding_scale
    x1p, y1p = cx - w_half, cy - h_half
    x2p, y2p = cx + w_half, cy + h_half

    # ── 6. 调整为原图宽高比（扩展短边，中心不变） ────────────────────────
    ar    = W / H
    w_cur = x2p - x1p
    h_cur = y2p - y1p

    if w_cur / max(h_cur, 1e-6) > ar:
        # 当前框相对"太宽" → 扩展高度
        h_new = w_cur / ar
        y1p   = cy - h_new / 2.0
        y2p   = cy + h_new / 2.0
    else:
        # 当前框相对"太高"（或恰好） → 扩展宽度
        w_new = h_cur * ar
        x1p   = cx - w_new / 2.0
        x2p   = cx + w_new / 2.0

    # ── 7. 平移使框落在图像范围内 ─────────────────────────────────────────
    x1c, y1c, x2c, y2c = _shift_bbox_into_image(x1p, y1p, x2p, y2p, W, H)
    bbox_crop = (x1c, y1c, x2c, y2c)

    # ── 8. Crop + BICUBIC resize ──────────────────────────────────────────
    crop_src = img_src.crop(bbox_crop).resize((W, H), Image.BICUBIC)
    crop_tgt = img_tgt.crop(bbox_crop).resize((W, H), Image.BICUBIC)

    meta = {
        "found_change":  True,
        "bbox_diff":     bbox_diff,
        "bbox_crop":     bbox_crop,
        "original_size": (W, H),
    }
    return crop_src, crop_tgt, meta
