import math
import os
import json
import pathlib
from collections import Counter
from datetime import datetime


# Load magic bytes mapping
MAGIC_BYTES_PATH = pathlib.Path(__file__).parent / "data" / "magic_bytes.json"

with open(MAGIC_BYTES_PATH, "r") as f:
    MAGIC_BYTES = json.load(f)


def file_entropy(filepath: str) -> float:
    """Calculate Shannon entropy of a file's byte content."""
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        if not data:
            return 0.0
        counts = Counter(data)
        total = len(data)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())
    except (PermissionError, OSError):
        return -1.0


def check_magic_bytes(filepath: str) -> bool:
    """
    Returns True if the file's magic bytes DON'T match its extension (mismatch = suspicious).
    Returns False if they match or if the extension is unknown.
    """
    ext = pathlib.Path(filepath).suffix.lower()
    if ext not in MAGIC_BYTES:
        return False
    expected_hex = MAGIC_BYTES[ext]
    expected_bytes = bytes.fromhex(expected_hex)
    try:
        with open(filepath, "rb") as f:
            header = f.read(len(expected_bytes))
        return header != expected_bytes
    except (PermissionError, OSError):
        return False


def is_executable(filepath: str) -> bool:
    """Returns True if the file has executable permission bits set."""
    try:
        mode = os.stat(filepath).st_mode
        return bool(mode & 0o111)
    except (PermissionError, OSError):
        return False


def get_file_size(filepath: str) -> int:
    """Returns file size in bytes."""
    try:
        return os.path.getsize(filepath)
    except (PermissionError, OSError):
        return -1


def get_modification_hour(filepath: str) -> int:
    """Returns the hour of day (0-23) the file was last modified."""
    try:
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime).hour
    except (PermissionError, OSError):
        return -1


def get_extension_size_zscore(filepath: str, size_stats: dict) -> float:
    """
    Returns how many standard deviations this file's size is from
    the mean size for its extension. Requires precomputed size_stats dict.
    """
    ext = pathlib.Path(filepath).suffix.lower()
    size = get_file_size(filepath)
    if ext not in size_stats or size == -1:
        return 0.0
    mean, std = size_stats[ext]
    if std == 0:
        return 0.0
    return (size - mean) / std


def extract_features(filepath: str, size_stats: dict = {}) -> dict:
    """
    Master function — extracts all features for a single file.
    Returns a dict of feature name -> value.
    """
    return {
        "filepath": filepath,
        "entropy": file_entropy(filepath),
        "magic_mismatch": int(check_magic_bytes(filepath)),
        "is_executable": int(is_executable(filepath)),
        "file_size": get_file_size(filepath),
        "modification_hour": get_modification_hour(filepath),
        "size_zscore": get_extension_size_zscore(filepath, size_stats),
    }