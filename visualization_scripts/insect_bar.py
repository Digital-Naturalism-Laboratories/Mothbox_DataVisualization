#!/usr/bin/env python3
"""
Insect Bar Visualizer
======================
Packs insect images (with transparent backgrounds) into a vertical bar,
filling from the bottom up — like a bar chart column made of real insects.

The canvas width is fixed by the user; the height grows to fit all insects.
Packing uses a shelf algorithm (left-to-right, bottom-to-top), with silhouette
masking so insects interlock along their actual edges.

Usage:
    python insect_bar.py
    python insect_bar.py --width 1200 --scale 0.4
    python insect_bar.py --cluster --sort-large-bottom
    python insect_bar.py --input /path/to/rembg --output my_bar.png

Requirements:
    pip install Pillow numpy scipy

Optional (for clustering):
    pip install torch torchvision timm hdbscan
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

# ── Optional perceptual clustering ───────────────────────────────────────────
try:
    import torch
    import torchvision.transforms as T
    import hdbscan
    import timm
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    print("⚠  torch / torchvision / timm / hdbscan not found — clustering disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# INPUT FOLDER  ← set this to your insects directory
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FOLDER = r"F:\Deployments\Panama\Hoya_163m_unrulyArao_2025-01-26\2025-01-27\patches\rembg"

# ── Configuration defaults ────────────────────────────────────────────────────
DEFAULT_WIDTH          = 2000    # canvas width in pixels (height is auto)
DEFAULT_SCALE          = 0.2     # scale applied to each insect before packing
DEFAULT_PADDING        = 2       # extra transparent pixels around each silhouette
ALPHA_THRESHOLD        = 60      # alpha below this → treated as transparent
SEED                   = 42
BACKGROUND_COLOR       = None    # None = transparent; or (255,255,255) for white
OUTLINE_ENABLED        = False
OUTLINE_THICKNESS      = 1.0     # multiplier of padding
USE_CLUSTERING         = True
CLUSTER_BATCH_SIZE     = 8
SORT_BY_SIZE           = True    # sort images by non-transparent pixel area
SORT_LARGE_BOTTOM      = True    # True → largest insects at the bottom of the bar


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers (identical API to the spiral packer)
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path, scale: float) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGBA")
        if scale != 1.0:
            w, h = img.size
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
            )
        return img
    except Exception as e:
        print(f"  ⚠  Could not load {path.name}: {e}")
        return None


def get_tight_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    """
    Return (r0, c0, r1, c1) — the tight bounding box of non-transparent pixels.
    Returns None if the image is fully transparent.
    """
    alpha = np.array(img.split()[3])
    rows  = np.any(alpha > ALPHA_THRESHOLD, axis=1)
    cols  = np.any(alpha > ALPHA_THRESHOLD, axis=0)
    if not rows.any():
        return None
    r0, r1 = int(np.argmax(rows)),  int(len(rows)  - 1 - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)),  int(len(cols)  - 1 - np.argmax(cols[::-1]))
    return r0, c0, r1 + 1, c1 + 1  # r1/c1 exclusive


def get_mask(img: Image.Image, padding: int = DEFAULT_PADDING) -> tuple[np.ndarray, int, int]:
    """
    Return (mask, row_offset, col_offset) where mask is cropped to the tight
    bounding box of non-transparent pixels (plus padding dilation).
    row_offset / col_offset are where the top-left of the mask sits within the
    original image, so the image can be alpha-composited at the correct position.
    """
    bbox = get_tight_bbox(img)
    if bbox is None:
        # Fully transparent — return a minimal mask
        return np.zeros((1, 1), dtype=bool), 0, 0

    r0, c0, r1, c1 = bbox

    # Add a margin so dilation doesn't eat into neighbouring content
    margin = padding + 1
    r0c = max(0, r0 - margin)
    c0c = max(0, c0 - margin)
    r1c = min(img.height, r1 + margin)
    c1c = min(img.width,  c1 + margin)

    alpha  = np.array(img.split()[3])[r0c:r1c, c0c:c1c]
    mask   = alpha > ALPHA_THRESHOLD
    if padding > 0:
        struct = np.ones((padding * 2 + 1, padding * 2 + 1), dtype=bool)
        mask   = binary_dilation(mask, structure=struct)
        # Trim dilation back to tight content (no extra empty border rows/cols)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any():
            tr0 = int(np.argmax(rows))
            tr1 = int(len(rows) - 1 - np.argmax(rows[::-1])) + 1
            tc0 = int(np.argmax(cols))
            tc1 = int(len(cols) - 1 - np.argmax(cols[::-1])) + 1
            mask  = mask[tr0:tr1, tc0:tc1]
            r0c  += tr0
            c0c  += tc0

    return mask, r0c, c0c


def get_image_area(path: Path) -> int:
    try:
        alpha = np.array(Image.open(path).convert("RGBA").split()[3])
        return int(np.sum(alpha > ALPHA_THRESHOLD))
    except Exception:
        return 0


def masks_overlap(canvas_occ: np.ndarray, insect_mask: np.ndarray,
                  row: int, col: int) -> bool:
    H, W   = canvas_occ.shape
    ih, iw = insect_mask.shape
    r0, c0 = max(0, row), max(0, col)
    r1, c1 = min(H, row + ih), min(W, col + iw)
    if r1 <= r0 or c1 <= c0:
        return True
    mr0, mc0 = r0 - row, c0 - col
    mr1, mc1 = r1 - row, c1 - col
    return bool(np.any(canvas_occ[r0:r1, c0:c1] & insect_mask[mr0:mr1, mc0:mc1]))


def stamp_mask(canvas_occ: np.ndarray, insect_mask: np.ndarray,
               row: int, col: int) -> None:
    H, W   = canvas_occ.shape
    ih, iw = insect_mask.shape
    r0, c0 = max(0, row), max(0, col)
    r1, c1 = min(H, row + ih), min(W, col + iw)
    mr0, mc0 = r0 - row, c0 - col
    mr1, mc1 = r1 - row, c1 - col
    canvas_occ[r0:r1, c0:c1] |= insect_mask[mr0:mr1, mc0:mc1]


def place_image(canvas_rgba: np.ndarray, img: Image.Image,
                row: int, col: int,
                img_row_off: int = 0, img_col_off: int = 0) -> None:
    """
    Alpha-composite img onto canvas at (row, col).
    img_row_off / img_col_off shift which part of img is read, so the full
    image (including transparent margins) is composited at the right position
    even though the mask was cropped to a tight bbox.
    The canvas position is adjusted so the insect appears at its natural location.
    """
    H, W   = canvas_rgba.shape[:2]
    ih, iw = img.height, img.width
    # canvas destination starts at (row - img_row_off, col - img_col_off)
    # because the tight mask's top-left is img_row_off pixels into the image
    dest_row = row - img_row_off
    dest_col = col - img_col_off
    r0, c0 = max(0, dest_row), max(0, dest_col)
    r1, c1 = min(H, dest_row + ih), min(W, dest_col + iw)
    if r1 <= r0 or c1 <= c0:
        return
    mr0, mc0 = r0 - dest_row, c0 - dest_col
    mr1, mc1 = r1 - dest_row, c1 - dest_col

    src = np.array(img)[mr0:mr1, mc0:mc1].astype(float)
    dst = canvas_rgba[r0:r1, c0:c1].astype(float)
    sa  = src[..., 3:4] / 255.0
    da  = dst[..., 3:4] / 255.0
    oa  = sa + da * (1 - sa)
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.where(oa > 0,
                       (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / oa, 0)
    res = np.zeros_like(dst)
    res[..., :3] = np.clip(rgb, 0, 255)
    res[...,  3] = np.clip(oa[..., 0] * 255, 0, 255)
    canvas_rgba[r0:r1, c0:c1] = res.astype(np.uint8)


def draw_outline(canvas_rgba: np.ndarray, mask: np.ndarray,
                 row: int, col: int, thickness: int, color: tuple) -> None:
    """row/col here are the canvas coords of the tight mask top-left."""
    struct       = np.ones((thickness * 2 + 1, thickness * 2 + 1), dtype=bool)
    outline_mask = binary_dilation(mask, structure=struct)
    H, W   = canvas_rgba.shape[:2]
    ih, iw = outline_mask.shape
    r0, c0 = max(0, row), max(0, col)
    r1, c1 = min(H, row + ih), min(W, col + iw)
    if r1 <= r0 or c1 <= c0:
        return
    mr0, mc0 = r0 - row, c0 - col
    mr1, mc1 = r1 - row, c1 - col
    region = outline_mask[mr0:mr1, mc0:mc1]
    dst    = canvas_rgba[r0:r1, c0:c1]
    paint  = region & (dst[..., 3] == 0)
    dst[paint] = [*color, 255]
    canvas_rgba[r0:r1, c0:c1] = dst


# ─────────────────────────────────────────────────────────────────────────────
# Core packing — shelf algorithm, builds upward, canvas grows as needed
# ─────────────────────────────────────────────────────────────────────────────

class BottomUpPacker:
    """
    Shelf packer that fills left-to-right in rows, bottom-to-top.
    The canvas starts with an estimated height and expands automatically
    (by prepending new rows at the top) whenever it runs out of space.
    """

    EXPAND_CHUNK = 5000  # rows added each time the canvas needs to grow

    def __init__(self, width: int, initial_height: int, bg: tuple | None = None):
        self.W   = width
        self.bg  = bg
        self._init_canvas(initial_height)

    def _init_canvas(self, height: int) -> None:
        self.H        = height
        self.canvas   = np.zeros((height, self.W, 4), dtype=np.uint8)
        self.occ      = np.zeros((height, self.W), dtype=bool)
        if self.bg is not None:
            self.canvas[:, :, :3] = self.bg
            self.canvas[:, :,  3] = 255
        self.shelf_top    = height   # current shelf's top row (canvas coords)
        self.shelf_x      = 0
        self.shelf_h      = 0
        self.highest_used = height   # topmost row with content

    def _expand(self) -> None:
        """Prepend EXPAND_CHUNK blank rows at the top of the canvas."""
        extra_canvas = np.zeros((self.EXPAND_CHUNK, self.W, 4), dtype=np.uint8)
        extra_occ    = np.zeros((self.EXPAND_CHUNK, self.W), dtype=bool)
        if self.bg is not None:
            extra_canvas[:, :, :3] = self.bg
            extra_canvas[:, :,  3] = 255
        self.canvas       = np.concatenate([extra_canvas, self.canvas], axis=0)
        self.occ          = np.concatenate([extra_occ,    self.occ],    axis=0)
        self.H           += self.EXPAND_CHUNK
        self.shelf_top   += self.EXPAND_CHUNK
        self.highest_used += self.EXPAND_CHUNK

    def _try_row(self, row: int, mask: np.ndarray) -> int | None:
        """Scan left-to-right on shelf row; return x on success, None if shelf full."""
        iw = mask.shape[1]
        x  = self.shelf_x
        while x + iw <= self.W:
            if not masks_overlap(self.occ, mask, row, x):
                return x
            x += 1
        return None

    def add(self, img: Image.Image,
            mask: np.ndarray, mask_row_off: int, mask_col_off: int,
            outline: bool, outline_px: int, rng: random.Random) -> bool:
        """
        mask       — tight-bbox boolean mask (already dilated by padding)
        mask_row_off / mask_col_off — pixel offset of mask top-left within img
        """
        ih, iw = mask.shape

        # Insect wider than the canvas — skip (can never fit)
        if iw > self.W:
            return False

        # Try current shelf; open a new one above if needed
        row = self.shelf_top - ih
        if row < 0:
            self._expand()
            row = self.shelf_top - ih

        x = self._try_row(row, mask)

        if x is None:
            # Advance to a new shelf above the current one
            self.shelf_top -= max(self.shelf_h, 1)
            self.shelf_x    = 0
            self.shelf_h    = 0
            row = self.shelf_top - ih
            if row < 0:
                self._expand()
                row = self.shelf_top - ih
            x = self._try_row(row, mask)
            if x is None:
                return False  # shouldn't happen after expansion, but be safe

        # Place it — mask canvas coords are (row, x); image is offset by mask offsets
        stamp_mask(self.occ, mask, row, x)
        if outline:
            color = (rng.randint(30, 255), rng.randint(30, 255), rng.randint(30, 255))
            draw_outline(self.canvas, mask, row, x, outline_px, color)
        place_image(self.canvas, img, row, x,
                    img_row_off=mask_row_off, img_col_off=mask_col_off)

        self.shelf_x      = x + iw
        self.shelf_h      = max(self.shelf_h, ih)
        self.highest_used = min(self.highest_used, row)
        return True

    def crop(self) -> np.ndarray:
        """Return only the filled portion (from topmost content row to bottom)."""
        return self.canvas[self.highest_used:, :]


# ─────────────────────────────────────────────────────────────────────────────
# Clustering (same helpers as the spiral packer)
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(image_files, batch_size=8):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
        model  = model.to(device).eval()
        tf = T.Compose([T.Resize(518), T.CenterCrop(518), T.ToTensor(),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        embs = []
        for i in range(0, len(image_files), batch_size):
            batch = []
            for p in image_files[i:i + batch_size]:
                try:
                    batch.append(tf(Image.open(p).convert("RGB")))
                except Exception:
                    batch.append(torch.zeros(3, 518, 518))
            with torch.no_grad():
                feats = model(torch.stack(batch).to(device))
            embs.extend(feats.cpu().numpy())
            print(f"  Embeddings: {min(i+batch_size, len(image_files))}/{len(image_files)}", end="\r")
        print()
        return np.array(embs)
    except Exception as e:
        print(f"  ⚠ DINOv2 unavailable ({e}), falling back to histogram embeddings.")
        result = []
        for p in image_files:
            try:
                img = np.array(Image.open(p).convert("RGB").resize((64, 64)),
                               dtype=np.float32) / 255.0
                hist = np.concatenate([
                    np.histogram(img[:, :, c], bins=32, range=(0, 1))[0].astype(np.float32)
                    for c in range(3)
                ])
                norm = np.linalg.norm(hist)
                result.append(hist / norm if norm > 0 else hist)
            except Exception:
                result.append(np.zeros(96, dtype=np.float32))
        return np.array(result)


def cluster_paths(paths, batch_size=8, vis_dir: Path = None):
    """
    Cluster images with HDBSCAN; return (sorted_paths, labels).
    Checks / saves a JSON cache in vis_dir.
    """
    import json
    from collections import defaultdict

    cache_path = vis_dir / "cluster_cache.json" if vis_dir else None

    # ── Try cache ──
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        path_strs = [str(p) for p in paths]
        if all(ps in cache for ps in path_strs):
            print(f"✓ Loaded cluster cache ({len(cache)} entries)")
            labels = [cache[ps] for ps in path_strs]
            return paths, labels
        print("⚠  Cache doesn't cover all images — re-clustering.")

    # ── Full clustering ──
    print(f"Extracting embeddings for {len(paths)} images…")
    embs = extract_embeddings([str(p) for p in paths], batch_size)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3, min_samples=1,
        cluster_selection_epsilon=0.05, metric="euclidean"
    )
    labels = clusterer.fit_predict(embs)
    n_c = len(set(labels) - {-1})
    print(f"✓ Found {n_c} perceptual clusters ({int(np.sum(labels==-1))} noise images)")

    if cache_path:
        vis_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({str(p): int(l) for p, l in zip(paths, labels)}, f, indent=2)
        print(f"✓ Cache saved → {cache_path}")

    return paths, list(labels)


def sort_by_cluster_then_size(paths, labels, large_bottom: bool, sort_by_size: bool = True):
    """
    Group images by cluster label; sort clusters by their representative area.
    large_bottom=True  → big clusters placed first (they land at the bottom of the bar).

    If sort_by_size is True:
      - Images within each named cluster are sorted by area (large_bottom order).
      - Noise images (label -1) are also sorted by area before being appended last.
    """
    from collections import defaultdict
    clusters = defaultdict(list)
    noise    = []
    for p, l in zip(paths, labels):
        if l == -1:
            noise.append(p)
        else:
            clusters[l].append(p)

    def rep_area(items):
        return get_image_area(items[0])

    # Sort clusters by the area of their representative (first) image
    ordered_clusters = sorted(clusters.values(), key=rep_area, reverse=large_bottom)

    result = []
    for cluster in ordered_clusters:
        if sort_by_size:
            cluster = sorted(cluster, key=get_image_area, reverse=large_bottom)
        result.extend(cluster)

    # Noise points last — sort by size too if requested
    if sort_by_size:
        noise = sorted(noise, key=get_image_area, reverse=large_bottom)
    result.extend(noise)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pack insect images into a vertical bar (fills bottom-up)."
    )
    parser.add_argument("--input",    "-i", default=INPUT_FOLDER)
    parser.add_argument("--output",   "-o", default="insect_bar.png")
    parser.add_argument("--width",    "-W", type=int,   default=DEFAULT_WIDTH,
                        help=f"Canvas width in pixels (height is auto). (default: {DEFAULT_WIDTH})")
    parser.add_argument("--scale",    "-s", type=float, default=DEFAULT_SCALE,
                        help=f"Scale factor for each insect image. (default: {DEFAULT_SCALE})")
    parser.add_argument("--padding",  "-p", type=int,   default=DEFAULT_PADDING,
                        help=f"Pixel gap around each silhouette. (default: {DEFAULT_PADDING})")
    parser.add_argument("--limit",    "-l", type=int,   default=None,
                        help="Max images to pack (default: all).")
    parser.add_argument("--seed",           type=int,   default=SEED)
    parser.add_argument("--background", "-b", default=None,
                        help="Background as R,G,B (e.g. 255,255,255). Default: transparent.")
    parser.add_argument("--outline",        action="store_true", default=OUTLINE_ENABLED,
                        help="Draw a random-coloured silhouette under each insect.")
    parser.add_argument("--outline-thickness", type=float, default=OUTLINE_THICKNESS)
    parser.add_argument("--cluster",        action="store_true", default=USE_CLUSTERING,
                        help="Cluster images perceptually so similar insects are grouped.")
    parser.add_argument("--sort-by-size",   action=argparse.BooleanOptionalAction,
                        default=SORT_BY_SIZE,
                        help="Sort images by area (non-transparent pixels). (default: on)")
    parser.add_argument("--sort-large-bottom", action=argparse.BooleanOptionalAction,
                        default=SORT_LARGE_BOTTOM,
                        help="Largest insects at the bottom of the bar. (default: on)")
    parser.add_argument("--no-shuffle",     action="store_true",
                        help="Disable default random shuffling (ignored if --cluster set).")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ── Collect paths ──────────────────────────────────────────────────────────
    input_dir = Path(args.input)
    exts      = {".png", ".webp", ".tif", ".tiff", ".jpg", ".jpeg"}
    paths     = sorted([p for p in input_dir.rglob("*")
                        if p.suffix.lower() in exts
                        and "visualizations" not in p.parts])
    if not paths:
        print(f"No images found in {input_dir}")
        return

    vis_dir = input_dir / "visualizations"

    # ── Ordering ───────────────────────────────────────────────────────────────
    if args.cluster:
        if CLUSTERING_AVAILABLE:
            print("Clustering images…")
            paths, labels = cluster_paths(paths, CLUSTER_BATCH_SIZE, vis_dir)
            paths = sort_by_cluster_then_size(paths, labels, args.sort_large_bottom, sort_by_size=args.sort_by_size)
        else:
            print("⚠  Clustering deps missing — using random order.")
            rng.shuffle(paths)
    else:
        if not args.no_shuffle:
            rng.shuffle(paths)

    # ── Sort by individual area ────────────────────────────────────────────────
    if args.sort_by_size and not args.cluster:
        print("Sorting images by area…")
        paths = sorted(paths, key=get_image_area, reverse=args.sort_large_bottom)
        print(f"  → {'largest first (bottom)' if args.sort_large_bottom else 'smallest first (bottom)'}")

    if args.limit:
        paths = paths[:args.limit]

    print(f"Found {len(paths)} images  |  width={args.width}px  |  scale={args.scale}  |  bottom-up bar")

    # ── Canvas setup ───────────────────────────────────────────────────────────
    # Sample images to estimate initial canvas height. The packer expands
    # automatically (in 5000-row chunks) if the estimate is too small, so
    # every insect will always find a home.
    bg = args.background or BACKGROUND_COLOR
    if isinstance(bg, str):
        bg = tuple(int(x) for x in bg.split(","))

    sample = paths[:min(50, len(paths))]
    h_samples, w_samples = [], []
    for sp in sample:
        try:
            w, h = Image.open(sp).size
            h_samples.append(int(h * args.scale))
            w_samples.append(int(w * args.scale))
        except Exception:
            pass
    if h_samples:
        avg_h = int(np.mean(h_samples))
        avg_w = int(np.mean(w_samples))
        insects_per_row = max(1, args.width // max(avg_w, 1))
        n_rows_est      = (len(paths) // insects_per_row) + 1
        initial_h       = avg_h * n_rows_est * 2   # ×2 slack for irregular shapes
        initial_h       = max(initial_h, 4000)
    else:
        initial_h = 20000

    print(f"  Initial canvas height estimate: {initial_h}px (auto-expands if needed)")
    packer = BottomUpPacker(args.width, initial_h, bg=bg)

    outline_px = max(1, int(args.padding * args.outline_thickness))

    # ── Pack ───────────────────────────────────────────────────────────────────
    placed  = 0
    invalid = 0
    no_fit  = 0
    t0      = time.time()

    for idx, path in enumerate(paths):
        img = load_image(path, args.scale)
        if img is None:
            invalid += 1
            continue

        mask, mask_row_off, mask_col_off = get_mask(img, padding=args.padding)
        if not mask.any():
            invalid += 1
            continue

        ok = packer.add(img, mask, mask_row_off, mask_col_off,
                        outline=args.outline,
                        outline_px=outline_px,
                        rng=rng)
        if ok:
            placed += 1
        else:
            no_fit += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(paths):
            elapsed = time.time() - t0
            pct     = 100 * (idx + 1) / len(paths)
            print(f"  [{idx+1:>5}/{len(paths)}]  {pct:5.1f}%  "
                  f"placed={placed}  invalid={invalid}  no_fit={no_fit}  "
                  f"elapsed={elapsed:.1f}s")

    # ── Crop & save ────────────────────────────────────────────────────────────
    result = packer.crop()
    if bg is not None:
        # Re-fill background on the cropped canvas (cropped region may have zeros)
        blank = result[..., 3] == 0
        result[blank, :3] = bg
        result[blank,  3] = 255

    vis_dir.mkdir(parents=True, exist_ok=True)

    cluster_tag = "_clustered" if args.cluster else ""
    size_tag    = f"_{'largebottom' if args.sort_large_bottom else 'smallbottom'}" if args.sort_by_size else ""
    suffix      = f"_w{args.width}_s{args.scale}_p{args.padding}{cluster_tag}{size_tag}"

    out_path = Path(args.output)
    out_path = vis_dir / out_path.with_stem(out_path.stem + suffix).name

    final_h = result.shape[0]
    Image.fromarray(result, "RGBA").save(out_path, "PNG")

    print(f"\n✓ Saved → {out_path}")
    print(f"  Canvas: {args.width} × {final_h} px  (height auto)")
    print(f"  placed={placed}  |  invalid/transparent={invalid}  |  couldn't fit={no_fit}")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()