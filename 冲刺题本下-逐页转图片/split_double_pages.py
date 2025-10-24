#!/usr/bin/env python3
"""
Automatic splitter for double-page book scans.

The script analyses each image in a directory, identifies the gutter by
measuring text density in a central column range, and saves the left/right
pages as separate files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    import cv2  # type: ignore[import]
except ImportError:  # pragma: no cover - fallback path
    cv2 = None

from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ADAPTIVE_BLOCK_SIZE = 35
ADAPTIVE_C = 15


def read_image(path: Path) -> np.ndarray:
    """Load an image as a BGR numpy array."""
    if cv2 is not None:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        array = np.array(rgb, dtype=np.uint8)
    return array[:, :, ::-1].copy()


def write_image(path: Path, image: np.ndarray) -> bool:
    """Persist a BGR numpy array to disk."""
    if cv2 is not None:
        return bool(cv2.imwrite(str(path), image))

    rgb = image[:, :, ::-1]
    Image.fromarray(rgb).save(path)
    return True


def preprocess_for_projection(image: np.ndarray) -> np.ndarray:
    """Return a binary image emphasising text regions for projection analysis."""
    if cv2 is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=ADAPTIVE_BLOCK_SIZE,
            C=ADAPTIVE_C,
        )
        return binary

    gray = (
        image[:, :, 0] * 0.114
        + image[:, :, 1] * 0.587
        + image[:, :, 2] * 0.299
    ).astype(np.float32)
    blurred = gaussian_filter(gray, sigma=1.2, mode="reflect")
    local_mean = uniform_filter(blurred, size=ADAPTIVE_BLOCK_SIZE, mode="reflect")
    threshold = local_mean - ADAPTIVE_C
    binary = np.where(blurred <= threshold, 255, 0).astype(np.uint8)
    return binary


def find_gutter_column(
    binary: np.ndarray,
    search_region: Tuple[float, float] = (0.35, 0.65),
    min_valley_ratio: float = 0.02,
) -> int:
    """
    Detect the x-coordinate of the gutter using vertical projection analysis.

    The detection focuses on a central band to avoid margins and uses a moving
    average to find a stable minimum in text density.
    """
    height, width = binary.shape[:2]

    start = max(int(width * search_region[0]), 0)
    end = min(int(width * search_region[1]), width)
    if end <= start:
        start = width // 3
        end = width - start

    projection = (binary // 255).sum(axis=0).astype(np.float32)

    window = max(int(width * min_valley_ratio), 15)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(projection, kernel, mode="same")

    search_slice = smoothed[start:end]
    valley_index = int(np.argmin(search_slice))
    x_cut = start + valley_index

    x_cut = int(np.clip(x_cut, 1, width - 2))
    return x_cut


def process_image(path: Path, output_dir: Path, overwrite: bool = False) -> Tuple[Path, Path]:
    """Split a single image and persist the left/right pages."""
    image = read_image(path)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    binary = preprocess_for_projection(image)
    x_cut = find_gutter_column(binary)

    height, width = image.shape[:2]
    if x_cut <= 0 or x_cut >= width:
        x_cut = width // 2

    left_page = image[:, :x_cut]
    right_page = image[:, x_cut:]

    if left_page.size == 0 or right_page.size == 0:
        raise RuntimeError(f"Failed to split image cleanly: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix
    left_path = output_dir / f"{path.stem}_left{suffix}"
    right_path = output_dir / f"{path.stem}_right{suffix}"

    if not overwrite and left_path.exists():
        raise FileExistsError(f"Output already exists: {left_path}")
    if not overwrite and right_path.exists():
        raise FileExistsError(f"Output already exists: {right_path}")

    if not write_image(left_path, left_page):
        raise RuntimeError(f"Failed to write file: {left_path}")
    if not write_image(right_path, right_page):
        raise RuntimeError(f"Failed to write file: {right_path}")

    return left_path, right_path


def find_images(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield image files from the given iterable that match supported extensions."""
    for path in paths:
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split double-page scans into single pages by locating the gutter."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing double-page images (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store split pages (default: same as input).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files when set.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of file extensions to process (default: common image types).",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    input_dir = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve() if args.output_dir else input_dir

    extensions = (
        {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}
        if args.extensions
        else SUPPORTED_EXTENSIONS
    )

    image_paths = sorted(find_images(input_dir.iterdir()), key=lambda p: p.name)
    image_paths = [p for p in image_paths if p.suffix.lower() in extensions]

    if not image_paths:
        print(f"No matching images found in {input_dir}", file=sys.stderr)
        return 1

    total = len(image_paths)
    for idx, image_path in enumerate(image_paths, start=1):
        try:
            left_path, right_path = process_image(image_path, output_dir, overwrite=args.overwrite)
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}/{total}] Failed: {image_path.name}: {exc}", file=sys.stderr)
            continue
        print(f"[{idx}/{total}] Split {image_path.name} -> {left_path.name}, {right_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
