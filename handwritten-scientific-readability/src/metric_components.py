"""Report the raw values behind each readability metric.

The scores in ``document_readability`` are each scaled against a fixed
reference value. This script prints, for every page, the raw measurement,
the reference value it is scaled against, and the resulting score, so the
numbers can be reported in a paper table. It recomputes exactly the same
quantities as ``document_readability`` and does not change any score.

Usage:
    python src/metric_components.py notebookimages/neat.png notebookimages/average.png ...
    python src/metric_components.py --image-folder notebookimages --output-csv metric_components.csv
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from src import document_readability as dr
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src import document_readability as dr

SIGMA_MAX_DEG = 15.0  # reference for baseline consistency
DENSITY_SCALE = 40.0  # reference multiplier for component density
SMALL_AREA_PX = 150  # components below this area count as "small"
NOISE_AREA_PX = 15  # components below this area are ignored


def spatial_components(binary: Any) -> Dict[str, float]:
    height = binary.shape[0]
    projection = np.sum(binary > 0, axis=1).tolist()
    max_projection = max(projection) if projection else 0
    var_max = float(height ** 2 / 16.0)
    out: Dict[str, float] = {
        "s_var_max_px2": var_max,
        "s_reference_sd_px": math.sqrt(var_max),
    }
    if max_projection == 0:
        return out
    rows = [i for i, v in enumerate(projection) if v / max_projection > 0.15]
    groups = []
    start = prev = rows[0]
    for idx in rows[1:]:
        if idx - prev > 1:
            groups.append((start, prev))
            start = idx
        prev = idx
    groups.append((start, prev))
    centers = [(s + e) / 2.0 for s, e in groups]
    spacings = [b - a for a, b in zip(centers, centers[1:])]
    var_d = statistics.pvariance(spacings) if len(spacings) > 1 else 0.0
    out.update(
        {
            "s_num_bands": float(len(centers)),
            "s_mean_spacing_px": float(np.mean(spacings)) if spacings else 0.0,
            "s_spacing_var_px2": float(var_d),
            "s_spacing_sd_px": math.sqrt(var_d),
        }
    )
    return out


def fragmentation_components(binary: Any) -> Dict[str, float]:
    cv2 = dr._ensure_cv2()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, _, stats, cents = cv2.connectedComponentsWithStats(closed, connectivity=8)
    areas, pts = [], []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < NOISE_AREA_PX:
            continue
        areas.append(float(a))
        pts.append(cents[i])
    page_h, page_w = closed.shape
    page_scale = math.sqrt(page_h * page_w)
    pts_arr = np.asarray(pts, dtype=np.float32)
    nn = []
    for i in range(len(pts_arr)):
        dist = np.sqrt(np.sum((pts_arr - pts_arr[i]) ** 2, axis=1))
        dist[i] = np.inf
        nn.append(float(np.min(dist)))

    count = len(areas)
    ink = sum(areas)
    density_raw = count / max(ink, 1)
    avg_nn = float(np.mean(nn)) if nn else 0.0
    small_ratio = sum(1 for a in areas if a < SMALL_AREA_PX) / max(count, 1)

    density_scaled = min(density_raw * DENSITY_SCALE, 1.0)
    dispersion_scaled = min(avg_nn / page_scale, 1.0)
    small_scaled = min(small_ratio, 1.0)
    index = 0.45 * density_scaled + 0.35 * dispersion_scaled + 0.20 * small_scaled
    return {
        "f_num_components": float(count),
        "f_ink_area_px": ink,
        "f_mean_component_area_px": ink / max(count, 1),
        "f_density_raw": density_raw,
        "f_density_scaled": density_scaled,
        "f_avg_nn_px": avg_nn,
        "f_nn_reference_px": page_scale,
        "f_dispersion_scaled": dispersion_scaled,
        "f_small_ratio": small_scaled,
        "f_contrib_density": 0.45 * density_scaled,
        "f_contrib_dispersion": 0.35 * dispersion_scaled,
        "f_contrib_small": 0.20 * small_scaled,
        "f_fragmentation_index": index,
        "f_score": max(0.0, min(1.0, 1.0 - index)),
    }


def page_components(image_path: str) -> Dict[str, Any]:
    _, binary, _ = dr.preprocess_image(image_path)
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = dr.compute_metrics(image_path)
    b = metrics.baseline_consistency
    row: Dict[str, Any] = {
        "image": Path(image_path).name,
        "height_px": binary.shape[0],
        "width_px": binary.shape[1],
        "b_score": b,
        "b_reference_deg": SIGMA_MAX_DEG,
        # b = 1 - sigma/15 (clipped), so sigma is recovered exactly unless clipped at 0
        "b_sigma_theta_deg": (1.0 - b) * SIGMA_MAX_DEG,
        "s_score": metrics.spatial_organization,
        "combined_score": metrics.combined_score,
    }
    row.update(spatial_components(binary))
    row.update(fragmentation_components(binary))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="*")
    parser.add_argument("--image-folder", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    images: List[str] = list(args.images)
    if args.image_folder:
        images += [str(p) for p in sorted(Path(args.image_folder).glob("*.png"))]
    if not images:
        parser.error("give image paths or --image-folder")

    rows = [page_components(p) for p in images]
    import pandas as pd

    df = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.set_index("image").T.to_string())
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
