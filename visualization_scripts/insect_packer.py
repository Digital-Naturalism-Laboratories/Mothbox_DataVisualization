#!/usr/bin/env python3
"""
Insect Shape Packing Visualizer
================================
Places insect images (with transparent backgrounds) onto a large canvas,
packing them tightly together using their actual silhouette edges.

Usage:
    python insect_packer.py --input ./insects --output packed.png
    python insect_packer.py --input ./insects --output packed.png --canvas 4000 --scale 0.5

Requirements:
    pip install Pillow numpy scipy
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

# ─────────────────────────────────────────────
# INPUT FOLDER  ← set this to your insects directory
# ─────────────────────────────────────────────
INPUT_FOLDER = r"C:\Users\andre\Desktop\Clear_Camilo_Bugs_BCI_Amour_Rainy_2025\rembg"

# ─────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────
DEFAULT_CANVAS  = 6000      # canvas width & height in pixels
DEFAULT_SCALE   = 1.0       # scale factor applied to each insect image
DEFAULT_PADDING = 3         # extra transparent pixels around each insect mask
MAX_ATTEMPTS    = 500       # placement attempts per image before giving up
ALPHA_THRESHOLD = 20        # alpha value below this → transparent (background)
SEED            = 42
OUTLINE_ENABLED   = True        # draw a coloured outline around each insect
OUTLINE_MODE      = "both"      # "outline_only" | "photo_only" | "both"
OUTLINE_THICKNESS = 5.0         # multiplier on padding for stroke width (1.0 = same as padding)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def draw_outline(canvas_rgba: np.ndarray,
                 mask: np.ndarray,
                 row: int, col: int,
                 thickness: int,
                 color: tuple) -> None:
    """
    Draw a filled silhouette (dilated mask) in a random colour underneath
    the insect, acting as a coloured outline/stroke.
    """
    from scipy.ndimage import binary_dilation
    struct       = np.ones((thickness * 2 + 1, thickness * 2 + 1), dtype=bool)
    outline_mask = binary_dilation(mask, structure=struct)

    H, W   = canvas_rgba.shape[:2]
    ih, iw = outline_mask.shape
    r0 = max(0, row);       c0 = max(0, col)
    r1 = min(H, row + ih);  c1 = min(W, col + iw)
    mr0 = r0 - row;  mc0 = c0 - col
    mr1 = r1 - row;  mc1 = c1 - col

    region = outline_mask[mr0:mr1, mc0:mc1]
    dst    = canvas_rgba[r0:r1, c0:c1]

    # Only paint where outline exists but destination is transparent
    paint = region & (dst[..., 3] == 0)
    dst[paint] = [*color, 255]
    canvas_rgba[r0:r1, c0:c1] = dst


def load_image(path: Path, scale: float) -> Image.Image | None:
    """Load a PNG/WebP/TIFF with alpha channel; resize if needed."""
    try:
        img = Image.open(path).convert("RGBA")
        if scale != 1.0:
            w, h = img.size
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                              Image.LANCZOS)
        return img
    except Exception as e:
        print(f"  ⚠  Could not load {path.name}: {e}")
        return None


def get_mask(img: Image.Image, padding: int = DEFAULT_PADDING) -> np.ndarray:
    """
    Return a boolean mask (True = insect pixel) derived from the alpha channel.
    A small dilation adds a 'padding' buffer so insects don't literally touch.
    """
    alpha = np.array(img.split()[3])          # shape (H, W)
    mask  = alpha > ALPHA_THRESHOLD
    if padding > 0:
        struct = np.ones((padding * 2 + 1, padding * 2 + 1), dtype=bool)
        mask   = binary_dilation(mask, structure=struct)
    return mask


def bounding_box(mask: np.ndarray):
    """Return (row_min, col_min, row_max, col_max) of the True region."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax


def masks_overlap(canvas_occ: np.ndarray,
                  insect_mask: np.ndarray,
                  row: int, col: int) -> bool:
    """
    Check whether placing insect_mask at (row, col) on canvas_occ
    would overlap any already-occupied pixel.
    """
    H, W   = canvas_occ.shape
    ih, iw = insect_mask.shape

    # Compute intersection between canvas area and insect mask
    r0 = max(0, row);       c0 = max(0, col)
    r1 = min(H, row + ih);  c1 = min(W, col + iw)
    if r1 <= r0 or c1 <= c0:
        return True   # fully outside → treat as overlap (don't place off-canvas)

    mr0 = r0 - row;  mc0 = c0 - col
    mr1 = r1 - row;  mc1 = c1 - col

    canvas_region = canvas_occ[r0:r1, c0:c1]
    insect_region = insect_mask[mr0:mr1, mc0:mc1]

    return bool(np.any(canvas_region & insect_region))


def stamp_mask(canvas_occ: np.ndarray,
               insect_mask: np.ndarray,
               row: int, col: int) -> None:
    """Write insect_mask into canvas_occ at (row, col)."""
    H, W   = canvas_occ.shape
    ih, iw = insect_mask.shape
    r0 = max(0, row);       c0 = max(0, col)
    r1 = min(H, row + ih);  c1 = min(W, col + iw)
    mr0 = r0 - row;  mc0 = c0 - col
    mr1 = r1 - row;  mc1 = c1 - col
    canvas_occ[r0:r1, c0:c1] |= insect_mask[mr0:mr1, mc0:mc1]


def place_image(canvas_rgba: np.ndarray,
                img: Image.Image,
                row: int, col: int) -> None:
    """Alpha-composite img onto canvas_rgba at (row, col)."""
    H, W   = canvas_rgba.shape[:2]
    ih, iw = img.height, img.width
    r0 = max(0, row);       c0 = max(0, col)
    r1 = min(H, row + ih);  c1 = min(W, col + iw)
    mr0 = r0 - row;  mc0 = c0 - col
    mr1 = r1 - row;  mc1 = c1 - col

    src  = np.array(img)[mr0:mr1, mc0:mc1].astype(float)   # (h, w, 4)
    dst  = canvas_rgba[r0:r1, c0:c1].astype(float)

    sa = src[..., 3:4] / 255.0
    da = dst[..., 3:4] / 255.0
    out_a = sa + da * (1 - sa)

    with np.errstate(invalid='ignore', divide='ignore'):
        out_rgb = np.where(
            out_a > 0,
            (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / out_a,
            0
        )

    result       = np.zeros_like(dst)
    result[..., :3] = np.clip(out_rgb, 0, 255)
    result[...,  3] = np.clip(out_a[..., 0] * 255, 0, 255)
    canvas_rgba[r0:r1, c0:c1] = result.astype(np.uint8)



# ─────────────────────────────────────────────
# Placement strategy: spiral outward from centre
# ─────────────────────────────────────────────

def spiral_positions(cx: int, cy: int, max_r: int, step: int = 6):
    """
    Yield (row, col) candidate positions in an Archimedean spiral
    outward from (cy, cx).  step controls density of the spiral.
    """
    r = 0
    theta = 0.0
    while r < max_r:
        row = int(cy + r * np.sin(theta))
        col = int(cx + r * np.cos(theta))
        yield row, col
        theta += step / max(r, 1)
        r      = step * theta / (2 * np.pi)


def try_place(canvas_occ: np.ndarray,
              canvas_rgba: np.ndarray,
              img: Image.Image,
              mask: np.ndarray,
              cx: int, cy: int,
              max_attempts: int,
              rng: random.Random,
              outline: bool = False,
              outline_thickness: int = 2,
              outline_mode: str = "both") -> bool:
    """
    Try to place img/mask near the canvas centre using a jittered spiral.
    Returns True on success.
    """
    H, W   = canvas_occ.shape
    ih, iw = mask.shape
    # offset so the insect *centre* lands on the spiral point
    ro, co = ih // 2, iw // 2

    max_r = max(H, W)
    step  = max(4, min(ih, iw) // 4)

    attempts = 0
    for (sr, sc) in spiral_positions(cx, cy, max_r, step=step):
        if attempts >= max_attempts:
            break
        attempts += 1

        # Small random jitter so identical sizes don't stack
        jr = rng.randint(-step, step)
        jc = rng.randint(-step, step)
        row = sr - ro + jr
        col = sc - co + jc

        if not masks_overlap(canvas_occ, mask, row, col):
            stamp_mask(canvas_occ, mask, row, col)
            if outline:
                color = (rng.randint(30,255), rng.randint(30,255), rng.randint(30,255))
                draw_outline(canvas_rgba, mask, row, col, outline_thickness, color)
            if outline_mode != "outline_only":
                place_image(canvas_rgba, img, row, col)
            return True


    return False


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pack insect images onto a transparent canvas using shape packing."
    )
    parser.add_argument("--outline",           action="store_true", default=OUTLINE_ENABLED,
                    help="Draw a random-coloured outline around each insect.")
    parser.add_argument("--outline-mode",      default=OUTLINE_MODE,
                        choices=["both", "outline_only", "photo_only"],
                        help="What to render. (default: both)")
    parser.add_argument("--outline-thickness", type=float, default=OUTLINE_THICKNESS,
                        help="Stroke width as a multiplier of padding. (default: 1.0)")
    parser.add_argument("--input",    "-i", default=INPUT_FOLDER,
                        help="Folder containing insect images. (default: INPUT_FOLDER global)")
    parser.add_argument("--output",   "-o", default="insect_packed.png",
                        help="Output PNG file path. (default: insect_packed.png)")
    parser.add_argument("--canvas",   "-c", type=int, default=DEFAULT_CANVAS,
                        help=f"Canvas size in pixels (square). (default: {DEFAULT_CANVAS})")
    parser.add_argument("--scale",    "-s", type=float, default=DEFAULT_SCALE,
                        help=f"Scale factor for input images. (default: {DEFAULT_SCALE})")
    parser.add_argument("--padding",  "-p", type=int, default=DEFAULT_PADDING,
                        help=f"Pixel gap between packed insects. (default: {DEFAULT_PADDING})")
    parser.add_argument("--attempts", "-a", type=int, default=MAX_ATTEMPTS,
                        help=f"Max placement attempts per insect. (default: {MAX_ATTEMPTS})")
    parser.add_argument("--limit",    "-l", type=int, default=None,
                        help="Max number of images to pack (default: all).")
    parser.add_argument("--seed",           type=int, default=SEED,
                        help=f"Random seed. (default: {SEED})")
    parser.add_argument("--shuffle",        action="store_true",
                        help="Shuffle image order before packing.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ── Collect image paths ──────────────────
    input_dir = Path(args.input)
    exts      = {".png", ".webp", ".tif", ".tiff", ".jpg", ".jpeg"}
    paths     = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in exts])
    if not paths:
        print(f"No images found in {input_dir}")
        return

    if args.shuffle:
        rng.shuffle(paths)

    if args.limit:
        paths = paths[: args.limit]

    print(f"Found {len(paths)} images  |  canvas {args.canvas}×{args.canvas}  |  scale {args.scale}")

    # ── Allocate canvas ──────────────────────
    C  = args.canvas
    cx = cy = C // 2

    canvas_rgba = np.zeros((C, C, 4), dtype=np.uint8)   # RGBA, all transparent
    canvas_occ  = np.zeros((C, C),    dtype=bool)        # occupancy map

    # ── Pack images ──────────────────────────
    placed = 0
    skipped = 0
    t0 = time.time()

    for idx, path in enumerate(paths):
        img  = load_image(path, args.scale)
        if img is None:
            skipped += 1
            continue

        mask = get_mask(img, padding=args.padding)

        # Skip fully-transparent images
        if not mask.any():
            skipped += 1
            continue

        ok = try_place(canvas_occ, canvas_rgba, img, mask,
                            cx, cy, args.attempts, rng,
                            outline=args.outline,
                            outline_thickness=max(1, int(args.padding * args.outline_thickness)),
                            outline_mode=args.outline_mode)

        if ok:
            placed += 1
        else:
            skipped += 1

        # Progress
        if (idx + 1) % 50 == 0 or (idx + 1) == len(paths):
            elapsed = time.time() - t0
            pct     = 100 * (idx + 1) / len(paths)
            print(f"  [{idx+1:>5}/{len(paths)}]  {pct:5.1f}%  "
                  f"placed={placed}  skipped={skipped}  "
                  f"elapsed={elapsed:.1f}s")

    # ── Save ─────────────────────────────────
    out_path = Path(args.output)
    mode_tag = "" if not args.outline else f"_{args.outline_mode}"
    suffix = f"_c{args.canvas}_s{args.scale}_p{args.padding}{mode_tag}"
    out_path = out_path.with_stem(out_path.stem + suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = Image.fromarray(canvas_rgba, "RGBA")
    result.save(out_path, "PNG")
    print(f"\n✓ Saved → {out_path}  ({placed} insects packed, {skipped} skipped)")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()