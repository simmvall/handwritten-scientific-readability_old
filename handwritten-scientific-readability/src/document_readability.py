from __future__ import annotations

import math
import os
import statistics
from cv2 import threshold
import numpy as np
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from jiwer import cer as jiwer_cer  # type: ignore
except Exception:  # pragma: no cover
    jiwer_cer = None


@dataclass
class ReadabilityMetrics:
    baseline_consistency: float
    spatial_organization: float
    text_fragmentation: float
    combined_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for structural metrics")
    return cv2


def _load_image(image: Union[str, os.PathLike, Any]) -> Any:
    if isinstance(image, (str, os.PathLike)):
        path = os.fspath(image)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        cv2_mod = _ensure_cv2()
        img = cv2_mod.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return img

    if hasattr(image, "shape"):
        return image

    raise TypeError("image must be a file path or an image array")


def preprocess_image(image: Union[str, os.PathLike, Any]) -> Tuple[Any, Any, Any]:
    cv2_mod = _ensure_cv2()
    img = _load_image(image)

    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2_mod.cvtColor(img, cv2_mod.COLOR_BGR2GRAY)

    blur = cv2_mod.GaussianBlur(gray, (3, 3), 0)
    binary = cv2_mod.adaptiveThreshold(
        blur,
        255,
        cv2_mod.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2_mod.THRESH_BINARY_INV,
        31,
        11,
    )
    
    kernel = cv2_mod.getStructuringElement(
        cv2_mod.MORPH_RECT,
        (3, 3),
    )

    binary = cv2_mod.morphologyEx(
        binary,
        cv2_mod.MORPH_OPEN,
        kernel,
        iterations=1,
    )
    
    #edge detection using canny 
    edges = cv2_mod.Canny(binary, 50, 150)
    
    return gray, binary, edges

def extract_text_lines(binary):

    cv2_mod = _ensure_cv2()

    # -----------------------------------
    # Connected Components
    # -----------------------------------

    num_labels, labels, stats, centroids = (
        cv2_mod.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    components = []

    for i in range(1, num_labels):

        area = stats[i, cv2_mod.CC_STAT_AREA]

        if area < 15:
            continue

        x = stats[i, cv2_mod.CC_STAT_LEFT]
        y = stats[i, cv2_mod.CC_STAT_TOP]
        w = stats[i, cv2_mod.CC_STAT_WIDTH]
        h = stats[i, cv2_mod.CC_STAT_HEIGHT]

        components.append(
            {
                "left": x,
                "right": x + w,
                "top": y,
                "bottom_y": y + h,
                "center_x": x + w / 2,
                "center_y": y + h / 2,
                "width": w,
                "height": h,
            }
        )

    if not components:
        return []

        # -----------------------------------
    # Horizontal Projection Profile
    # -----------------------------------

    projection = np.sum(binary > 0, axis=1).astype(np.float32)

    # Light smoothing to suppress noise while
    # preserving valleys between text lines.
    kernel = np.ones(5, dtype=np.float32) / 5
    projection = np.convolve(projection, kernel, mode="same")

    # -----------------------------------
    # Find valleys (local minima)
    # -----------------------------------

    max_val = np.max(projection)

    valley_threshold = 0.08 * max_val

    valleys = []

    for y in range(1, len(projection) - 1):

        if (
            projection[y] <= projection[y - 1]
            and projection[y] <= projection[y + 1]
            and projection[y] < valley_threshold
        ):
            valleys.append(y)

    # Merge valleys that are extremely close
    merged_valleys = []

    for v in valleys:

        if (
            not merged_valleys
            or v - merged_valleys[-1] > 8
        ):
            merged_valleys.append(v)

    # -----------------------------------
    # Convert valleys into text bands
    # -----------------------------------

    bands = []

    previous = 0

    for valley in merged_valleys:

        if valley - previous > 6:
            bands.append((previous, valley))

        previous = valley

    if len(projection) - previous > 6:
        bands.append((previous, len(projection) - 1))

    # -----------------------------------
    # Assign connected components
    # -----------------------------------

    lines = []

    median_height = np.median(
        [c["height"] for c in components]
    )

    for top, bottom in bands:

        line = [
            c
            for c in components
            if top <= c["center_y"] <= bottom
        ]

        if len(line) < 4:
            continue

        band_height = bottom - top

        # Reject obviously merged regions.
        if band_height > 4 * median_height:
            continue

        line.sort(key=lambda c: c["center_x"])

        lines.append(line)

    return lines

def baseline_consistency(image: Union[str, os.PathLike, Any]) -> float:
    """
    Estimates handwriting baseline consistency using connected-component
    grouping instead of Hough transforms.

    Returns
    -------
    float
        Baseline consistency score in [0,1].
    """

    cv2_mod = _ensure_cv2()

    _, binary, _ = preprocess_image(image)

    lines = extract_text_lines(binary)
    
    if len(lines) < 2:
        return 0.0


    for i, line in enumerate(lines):
        print(
            f"Line {i}: "
            f"{len(line)} components"
        )
        
    angles = []
    weights = []

    for line in lines:

        median_height = np.median(
            [c["height"] for c in line]
        )

        filtered = [
            c
            for c in line
            if c["height"] >= 0.75 * median_height
        ]

        if len(filtered) < 4:
            filtered = line

        points = np.array(
            [
                [
                    c["center_x"],
                    c["bottom_y"],
                ]
                for c in filtered
            ],
            dtype=np.float32,
        )

        if len(points) < 4:
            continue

        vx, vy, x0, y0 = cv2_mod.fitLine(

            points,

            cv2_mod.DIST_L2,

            0,

            0.01,

            0.01,
        )

        vx = vx.item()
        vy = vy.item()
        x0 = x0.item()
        y0 = y0.item()

        theta = np.degrees(np.arctan2(vy, vx))

        if theta < -45:
            theta += 180
        elif theta > 45:
            theta -= 180

        # -------------------------------------------------
        # Reject poor baseline fits
        # -------------------------------------------------

        distances = []

        for x, y in points:

            distance = abs(
                vy * (x - x0)
                - vx * (y - y0)
            )

            distances.append(distance)

        line_error = np.mean(distances)

        # Skip merged text bands or noisy regions
        if line_error > 8:
            continue

        angles.append(theta)
        weights.append(len(filtered))

    if len(angles) < 2:
        return 0.0

    angles = np.asarray(angles)
    weights = np.asarray(weights, dtype=np.float64)

    weighted_mean = np.average(
        angles,
        weights=weights,
    )

    sigma_theta = np.sqrt(
        np.average(
            (angles - weighted_mean) ** 2,
            weights=weights,
        )
    )
    print(f"σθ = {sigma_theta:.2f}°")

    sigma_max = 15.0

    score = 1.0 - (sigma_theta / sigma_max)

    score = np.clip(score, 0.0, 1.0)

    return float(score)


def spatial_organization(image: Union[str, os.PathLike, Any]) -> float:
    _, binary, _ = preprocess_image(image)

    height = binary.shape[0]
    projection = np.sum(binary > 0, axis=1).tolist()

    if not projection or max(projection) == 0:
        return 0.0

    max_projection = max(projection)
    normalized = [value / max_projection for value in projection]
    row_indices = [idx for idx, value in enumerate(normalized) if value > 0.15]

    if len(row_indices) < 2:
        return 0.0

    groups: List[Tuple[int, int]] = []
    start = row_indices[0]
    prev = row_indices[0]

    for idx in row_indices[1:]:
        if idx - prev > 1:
            groups.append((start, prev))
            start = idx
        prev = idx

    groups.append((start, prev))

    centers = [(s + e) / 2.0 for (s, e) in groups if e - s + 1 > 0]
    if len(centers) < 2:
        return 0.0

    spacings = [b - a for a, b in zip(centers, centers[1:])]
    var_d = statistics.pvariance(spacings) if len(spacings) > 1 else 0.0
    var_max = float(height ** 2 / 16.0)

    if var_max == 0:
        return 0.0

    score = 1.0 - (var_d / var_max)
    return float(max(0.0, min(1.0, score)))


def text_fragmentation(image: Union[str, os.PathLike, Any]) -> float:
    cv2_mod = _ensure_cv2()

    _, binary, _ = preprocess_image(image)

    # -----------------------------------
    # Morphological closing
    # Merge nearby handwriting into
    # coherent text regions.
    # -----------------------------------

    kernel = cv2_mod.getStructuringElement(
        cv2_mod.MORPH_RECT,
        (5, 5),
    )

    closed = cv2_mod.morphologyEx(
        binary,
        cv2_mod.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    # -----------------------------------
    # Connected Components
    # -----------------------------------

    num_labels, labels, stats, centroids = cv2_mod.connectedComponentsWithStats(
        closed,
        connectivity=8,
    )

    components = []

    for i in range(1, num_labels):

        area = stats[i, cv2_mod.CC_STAT_AREA]

        # Ignore tiny speckles/noise
        if area < 15:
            continue

        x = stats[i, cv2_mod.CC_STAT_LEFT]
        y = stats[i, cv2_mod.CC_STAT_TOP]
        w = stats[i, cv2_mod.CC_STAT_WIDTH]
        h = stats[i, cv2_mod.CC_STAT_HEIGHT]

        cx, cy = centroids[i]

        components.append(
            {
                "area": area,
                "bbox": (x, y, w, h),
                "centroid": (cx, cy),
            }
        )
    
    if len(components) <= 1:
        return 1.0

    page_h, page_w = closed.shape
    page_area = float(page_h * page_w)

    # -----------------------------------
    # Component Density
    # -----------------------------------

    total_ink_area = sum(c["area"] for c in components)

    component_density = len(components) / max(total_ink_area, 1)

    # -----------------------------------
    # Small Component Ratio
    # -----------------------------------

    small_components = sum(
        1
        for c in components
        if c["area"] < 150
    )

    small_ratio = small_components / len(components)

    # -----------------------------------
    # Spatial Dispersion
    # Average nearest-neighbor distance
    # -----------------------------------

    distances = []

    centroids = np.array(
        [c["centroid"] for c in components],
        dtype=np.float32,
    )

    for i in range(len(centroids)):

        diff = centroids - centroids[i]

        dist = np.sqrt(np.sum(diff ** 2, axis=1))

        dist[i] = np.inf

        distances.append(np.min(dist))

    avg_nn = float(np.mean(distances))

    dispersion = avg_nn / math.sqrt(page_area)

    # -----------------------------------
    # Weighted Fragmentation Index
    # -----------------------------------

    component_density = min(component_density * 40, 1.0)
    dispersion = min(dispersion, 1.0)
    small_ratio = min(small_ratio, 1.0)

    fragmentation = (
        0.45 * component_density +
        0.35 * dispersion +
        0.20 * small_ratio
    )

    score = 1.0 - fragmentation

    score = max(0.0, min(1.0, score))

    return float(score)


def compute_metrics(image: Union[str, os.PathLike, Any]) -> ReadabilityMetrics:
    b = baseline_consistency(image)
    s = spatial_organization(image)
    f = text_fragmentation(image)
    combined = (
    0.40 * b +
    0.20 * s +
    0.40 * f
    )
    return ReadabilityMetrics(
        baseline_consistency=b,
        spatial_organization=s,
        text_fragmentation=f,
        combined_score=combined,
    )


def classify_score(score: float, metric_name: str) -> str:
    if metric_name == "text_fragmentation":
        if score >= 0.80:
            return "Low Fragmentation (High Readability)"
        if score >= 0.50:
            return "Moderate Fragmentation"
        return "High Fragmentation (Low Readability)"

    if score >= 0.80:
        return "High"
    if score >= 0.50:
        return "Moderate"
    return "Low"


def ocr_text_and_confidence(image: Union[str, os.PathLike, Any]) -> Tuple[str, float]:
    if pytesseract is None:
        return "", 0.0

    img = _load_image(image)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    texts: List[str] = []
    confidences: List[float] = []

    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        cleaned = str(text).strip()
        if cleaned:
            texts.append(cleaned)

        try:
            value = float(conf)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            confidences.append(value)

    transcript = " ".join(texts)
    avg_conf = float(statistics.mean(confidences)) if confidences else 0.0
    return transcript, avg_conf


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    return previous[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    if jiwer_cer is not None:
        return float(jiwer_cer(reference, hypothesis))

    reference_norm = " ".join(str(reference).split())
    hypothesis_norm = " ".join(str(hypothesis).split())
    distance = _levenshtein_distance(reference_norm, hypothesis_norm)
    denominator = max(1, len(reference_norm))
    return distance / denominator


def validate_against_ground_truth(
    image: Union[str, os.PathLike, Any],
    ground_truth_text: Optional[str] = None,
) -> Dict[str, Any]:
    ocr_text, avg_conf = ocr_text_and_confidence(image)
    reference = "" if ground_truth_text is None else str(ground_truth_text)

    return {
        "ocr_text": ocr_text,
        "ocr_confidence": avg_conf,
        "cer": compute_cer(reference, ocr_text),
    }


def _default_stem(image: Union[str, os.PathLike, Any]) -> str:
    if isinstance(image, (str, os.PathLike)):
        name = os.path.splitext(os.path.basename(os.fspath(image)))[0]
        return name or "page"
    return "page"


def analyze_page(
    image: Union[str, os.PathLike, Any],
    ground_truth_text: Optional[str] = None,
    output_dir: Optional[str] = "preprocessed_outputs",
) -> Dict[str, Any]:
    metrics = compute_metrics(image)
    result = metrics.to_dict()

    result["baseline_consistency_class"] = classify_score(metrics.baseline_consistency, "baseline_consistency")
    result["spatial_organization_class"] = classify_score(metrics.spatial_organization, "spatial_organization")
    result["text_fragmentation_class"] = classify_score(metrics.text_fragmentation, "text_fragmentation")

    if output_dir is not None:
        result["preprocessed_paths"] = save_preprocessed_images(
            image=image,
            output_dir=output_dir,
            stem=_default_stem(image),
        )

    if ground_truth_text is not None:
        result.update(validate_against_ground_truth(image, ground_truth_text))

    return result


def batch_analyze(
    images: Iterable[Union[str, os.PathLike, Any]],
    ground_truth_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for image in images:
        image_key = os.fspath(image) if isinstance(image, (str, os.PathLike)) else str(id(image))
        ground_truth_text = None
        if ground_truth_map is not None:
            ground_truth_text = ground_truth_map.get(image_key)

        payload = analyze_page(image, ground_truth_text)
        payload["image_key"] = image_key
        rows.append(payload)

    return rows


def save_preprocessed_images(
    image: Union[str, os.PathLike, Any],
    output_dir: str,
    stem: str = "page",
) -> Dict[str, str]:
    cv2_mod = _ensure_cv2()
    os.makedirs(output_dir, exist_ok=True)

    gray, binary, edges = preprocess_image(image)

    # Same closing used by Text Fragmentation
    kernel = cv2_mod.getStructuringElement(
        cv2_mod.MORPH_RECT,
        (5, 5),
    )

    closed = cv2_mod.morphologyEx(
        binary,
        cv2_mod.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    # -----------------------------------
    # Connected components visualization
    # -----------------------------------

    num_labels, labels, stats, _ = cv2_mod.connectedComponentsWithStats(
        closed,
        connectivity=8,
    )

    component_vis = np.zeros(
        (closed.shape[0], closed.shape[1], 3),
        dtype=np.uint8,
    )

    rng = np.random.default_rng(42)

    for i in range(1, num_labels):

        area = stats[i, cv2_mod.CC_STAT_AREA]

        # Match the fragmentation metric
        if area < 50:
            continue

        color = rng.integers(
            50,
            255,
            size=3,
            dtype=np.uint8,
        )

        component_vis[labels == i] = color

    paths = {
        "gray": os.path.join(output_dir, f"{stem}_gray.png"),
        "binary": os.path.join(output_dir, f"{stem}_binary.png"),
        "edges": os.path.join(output_dir, f"{stem}_edges.png"),
        "closed": os.path.join(output_dir, f"{stem}_closed.png"),
        "components": os.path.join(output_dir, f"{stem}_components.png"),
    }

    cv2_mod.imwrite(paths["gray"], gray)
    cv2_mod.imwrite(paths["binary"], binary)
    cv2_mod.imwrite(paths["edges"], edges)
    cv2_mod.imwrite(paths["closed"], closed)
    cv2_mod.imwrite(paths["components"], component_vis)

    return paths
