import pathlib
import pandas as pd
from collections import defaultdict
from features import extract_features


SKIP_DIRS = {
    "venv", ".venv", "env", ".env",
    ".git", "__pycache__", "node_modules",
    ".Trash", "Trash"
}


def collect_file_paths(root_dir: str) -> list[str]:
    """
    Walk a directory recursively and return all file paths,
    skipping common noisy/irrelevant directories.
    """
    root = pathlib.Path(root_dir)
    paths = []

    for path in root.rglob("*"):
        # Skip directories themselves
        if not path.is_file():
            continue
        # Skip if any part of the path is in SKIP_DIRS
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        # Skip symlinks to avoid loops
        if path.is_symlink():
            continue
        paths.append(str(path))

    return paths


def compute_size_stats(paths: list[str]) -> dict:
    """
    For each extension, compute mean and std of file sizes.
    Used to calculate per-extension size z-scores in feature extraction.
    """
    size_by_ext = defaultdict(list)

    for filepath in paths:
        ext = pathlib.Path(filepath).suffix.lower()
        try:
            size = pathlib.Path(filepath).stat().st_size
            size_by_ext[ext].append(size)
        except (PermissionError, OSError):
            continue

    stats = {}
    for ext, sizes in size_by_ext.items():
        if len(sizes) < 2:
            continue
        mean = sum(sizes) / len(sizes)
        variance = sum((s - mean) ** 2 for s in sizes) / len(sizes)
        std = variance ** 0.5
        stats[ext] = (mean, std)

    return stats


def scan_directory(root_dir: str) -> pd.DataFrame:
    """
    Full scan pipeline:
    1. Collect all file paths
    2. Compute size stats for z-score normalisation
    3. Extract features for every file
    4. Return as a DataFrame
    """
    print(f"[scanner] Collecting file paths from: {root_dir}")
    paths = collect_file_paths(root_dir)
    print(f"[scanner] Found {len(paths)} files")

    print("[scanner] Computing size statistics per extension...")
    size_stats = compute_size_stats(paths)

    print("[scanner] Extracting features...")
    records = []
    for i, filepath in enumerate(paths):
        if i % 500 == 0 and i > 0:
            print(f"[scanner] Processed {i}/{len(paths)} files...")
        features = extract_features(filepath, size_stats)
        records.append(features)

    df = pd.DataFrame(records)
    print(f"[scanner] Done. DataFrame shape: {df.shape}")
    return df