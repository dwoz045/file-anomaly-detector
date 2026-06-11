import pandas as pd
import numpy as np
import json
import pathlib
import hashlib
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box


console = Console()

SNAPSHOTS_DIR = pathlib.Path(__file__).parent / "data" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Snapshot Management ──────────────────────────────────────────────────────

def snapshot_path(label: str) -> pathlib.Path:
    return SNAPSHOTS_DIR / f"{label}.parquet"


def save_snapshot(df: pd.DataFrame, label: str = None) -> str:
    """
    Save a scan result as a timestamped snapshot.
    Returns the label used to save it.
    """
    if label is None:
        label = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = snapshot_path(label)
    df.to_parquet(path, index=False)
    console.print(f"[dim][monitor] Snapshot saved: {path}[/dim]")
    return label


def load_snapshot(label: str) -> pd.DataFrame:
    """Load a previously saved snapshot by label."""
    path = snapshot_path(label)
    if not path.exists():
        raise FileNotFoundError(f"No snapshot found at {path}")
    return pd.read_parquet(path)


def list_snapshots() -> list[str]:
    """Return all saved snapshot labels sorted by time."""
    return sorted([p.stem for p in SNAPSHOTS_DIR.glob("*.parquet")])


def get_latest_snapshot() -> pd.DataFrame | None:
    """Load the most recent snapshot, or None if none exist."""
    snapshots = list_snapshots()
    if not snapshots:
        return None
    return load_snapshot(snapshots[-1])


# ─── File Hashing ─────────────────────────────────────────────────────────────

def hash_file(filepath: str, chunk_size: int = 65536) -> str:
    """
    Compute SHA256 hash of a file's contents.
    Used to detect files that have changed between scans.
    Returns empty string on permission error.
    """
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (PermissionError, OSError):
        return ""


def add_file_hashes(df: pd.DataFrame) -> pd.DataFrame:
    """Add a SHA256 hash column to a scan DataFrame."""
    console.print("[monitor] Computing file hashes...")
    df = df.copy()
    df["sha256"] = df["filepath"].apply(hash_file)
    return df


# ─── Change Detection ─────────────────────────────────────────────────────────

def diff_snapshots(old_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Compare two snapshots and return a dict of changes:

      - new_files:        files present in new scan but not old
      - deleted_files:    files present in old scan but not new
      - modified_files:   files present in both but with changed SHA256
      - entropy_spikes:   files whose entropy increased significantly
      - newly_flagged:    files not anomalous before but anomalous now
      - newly_critical:   files that moved from clean/low to both_flagged
    """
    old_paths = set(old_df["filepath"])
    new_paths = set(new_df["filepath"])

    new_files = new_df[new_df["filepath"].isin(new_paths - old_paths)].copy()
    deleted_files = old_df[old_df["filepath"].isin(old_paths - new_paths)].copy()

    # Files present in both scans
    common_paths = old_paths & new_paths
    old_common = old_df[old_df["filepath"].isin(common_paths)].set_index("filepath")
    new_common = new_df[new_df["filepath"].isin(common_paths)].set_index("filepath")

    # Hash-based modification detection
    if "sha256" in old_common.columns and "sha256" in new_common.columns:
        hash_changed = old_common["sha256"] != new_common["sha256"]
        modified_files = new_common[hash_changed].reset_index()
    else:
        modified_files = pd.DataFrame()

    # Entropy spike detection — flag files where entropy increased by > 1.5 bits
    # This is the ransomware signal: files being progressively encrypted
    if "entropy" in old_common.columns and "entropy" in new_common.columns:
        entropy_delta = new_common["entropy"] - old_common["entropy"]
        spike_mask = entropy_delta > 1.5
        entropy_spikes = new_common[spike_mask].copy().reset_index()
        entropy_spikes["entropy_delta"] = entropy_delta[spike_mask].values
    else:
        entropy_spikes = pd.DataFrame()

    # Newly flagged files — clean before, anomalous now
    if "is_anomaly" in old_common.columns and "is_anomaly" in new_common.columns:
        was_clean = old_common["is_anomaly"] == 0
        now_flagged = new_common["is_anomaly"] == 1
        newly_flagged = new_common[was_clean & now_flagged].reset_index()
    else:
        newly_flagged = pd.DataFrame()

    # Files that escalated to both_flagged consensus
    if "consensus" in old_common.columns and "consensus" in new_common.columns:
        was_not_critical = old_common["consensus"] != "both_flagged"
        now_critical = new_common["consensus"] == "both_flagged"
        newly_critical = new_common[was_not_critical & now_critical].reset_index()
    else:
        newly_critical = pd.DataFrame()

    return {
        "new_files": new_files,
        "deleted_files": deleted_files,
        "modified_files": modified_files,
        "entropy_spikes": entropy_spikes,
        "newly_flagged": newly_flagged,
        "newly_critical": newly_critical,
    }


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_diff_report(diff: dict, scan_time: float = None):
    """Print a rich formatted diff report between two snapshots."""
    console.print()
    console.print("[bold white]CHANGE DETECTION REPORT[/bold white]", justify="center")
    if scan_time:
        console.print(f"[dim]Scan completed in {scan_time:.1f}s[/dim]", justify="center")
    console.print()

    sections = [
        ("newly_critical",  "CRITICAL — Newly consensus-flagged files", "bold red"),
        ("entropy_spikes",  "HIGH     — Entropy spike detected (possible encryption)", "red"),
        ("newly_flagged",   "MEDIUM   — Newly anomalous files", "yellow"),
        ("modified_files",  "INFO     — Modified files (hash changed)", "cyan"),
        ("new_files",       "INFO     — New files detected", "cyan"),
        ("deleted_files",   "INFO     — Deleted files", "dim"),
    ]

    any_changes = False

    for key, label, colour in sections:
        df = diff.get(key, pd.DataFrame())
        if df is None or df.empty:
            continue

        any_changes = True
        console.print(f"[{colour}]{label}[/{colour}] ({len(df)} file(s))")

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
        table.add_column("File", overflow="fold")

        # Add relevant extra columns depending on section
        if key == "entropy_spikes" and "entropy_delta" in df.columns:
            table.add_column("Entropy Delta", width=14)
            for _, row in df.head(10).iterrows():
                table.add_row(row["filepath"], f"+{row['entropy_delta']:.2f}")

        elif key in ("newly_flagged", "newly_critical") and "anomaly_score" in df.columns:
            table.add_column("Anomaly Score", width=14)
            for _, row in df.head(10).iterrows():
                table.add_row(row["filepath"], f"{row['anomaly_score']:.4f}")

        elif key == "modified_files" and "entropy" in df.columns:
            table.add_column("Entropy", width=10)
            for _, row in df.head(10).iterrows():
                table.add_row(row["filepath"], f"{row['entropy']:.2f}")

        else:
            for _, row in df.head(10).iterrows():
                table.add_row(row["filepath"])

        console.print(table)

    if not any_changes:
        console.print("[green]No changes detected since last snapshot.[/green]")

    console.print()


# ─── Monitoring Loop ──────────────────────────────────────────────────────────

def run_monitor(
    directory: str,
    scan_fn,
    interval_seconds: int = 60,
    max_scans: int = None
):
    """
    Continuously scan a directory on a schedule and diff against
    the previous scan. Calls scan_fn(directory) which should return
    a fully scored DataFrame (IF + AE + consensus).

    interval_seconds: time between scans
    max_scans: stop after this many scans (None = run forever)
    """
    console.print(f"[bold green][monitor] Starting monitor on {directory}[/bold green]")
    console.print(f"[dim]Scan interval: {interval_seconds}s — press Ctrl+C to stop[/dim]\n")

    scan_count = 0
    previous_df = get_latest_snapshot()

    if previous_df is not None:
        console.print(f"[dim][monitor] Loaded previous snapshot ({list_snapshots()[-1]})[/dim]")

    try:
        while True:
            if max_scans and scan_count >= max_scans:
                console.print("[dim][monitor] Max scans reached. Stopping.[/dim]")
                break

            start = time.time()
            console.print(f"\n[dim][monitor] Running scan {scan_count + 1}...[/dim]")

            current_df = scan_fn(directory)
            current_df = add_file_hashes(current_df)
            elapsed = time.time() - start

            label = save_snapshot(current_df)

            if previous_df is not None:
                diff = diff_snapshots(previous_df, current_df)
                print_diff_report(diff, scan_time=elapsed)
            else:
                console.print("[dim][monitor] No previous snapshot to diff against — baseline established.[/dim]")

            previous_df = current_df
            scan_count += 1

            if max_scans is None or scan_count < max_scans:
                console.print(f"[dim][monitor] Next scan in {interval_seconds}s...[/dim]")
                time.sleep(interval_seconds)

    except KeyboardInterrupt:
        console.print("\n[dim][monitor] Stopped by user.[/dim]")