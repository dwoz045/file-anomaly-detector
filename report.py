import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
from pathlib import Path


console = Console()


def severity_label(score: float) -> tuple[str, str]:
    """
    Convert a raw anomaly score to a human-readable severity label and colour.
    Isolation Forest scores: more negative = more anomalous.
    Typical range is roughly -0.5 to +0.5.
    """
    if score < -0.15:
        return "HIGH", "red"
    elif score < -0.05:
        return "MEDIUM", "yellow"
    else:
        return "LOW", "cyan"


def explain_file(row: pd.Series) -> str:
    """
    Generate a human-readable explanation of why a file was flagged,
    based on which features are most anomalous.
    """
    reasons = []

    if row["entropy"] > 7.5:
        reasons.append(f"very high entropy ({row['entropy']:.2f}) — possible encryption/compression")
    elif row["entropy"] > 6.5:
        reasons.append(f"elevated entropy ({row['entropy']:.2f})")

    if row["magic_mismatch"] == 1:
        reasons.append("magic bytes don't match file extension")

    if row["is_executable"] == 1:
        reasons.append("executable permissions set")

    if abs(row["size_zscore"]) > 2:
        reasons.append(f"unusual size for its extension (z-score: {row['size_zscore']:.1f})")

    if not reasons:
        reasons.append("statistical outlier across combined features")

    return ", ".join(reasons)


def print_report(results: pd.DataFrame, top_n: int = 20):
    """
    Print a rich formatted report of the top N most anomalous files.
    """
    # Filter to flagged files only, sorted by anomaly score ascending (most anomalous first)
    flagged = results[results["is_anomaly"] == 1].sort_values("anomaly_score")

    total_scanned = len(results)
    total_flagged = len(flagged)

    # Header
    console.print()
    console.print("[bold white]FILE ANOMALY DETECTOR[/bold white]", justify="center")
    console.print(f"[dim]Scanned {total_scanned} files — {total_flagged} flagged as anomalous[/dim]", justify="center")
    console.print()

    if flagged.empty:
        console.print("[green]No anomalies detected.[/green]")
        return

    # Build table
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        expand=True
    )

    table.add_column("Severity", width=8)
    table.add_column("Entropy", width=8)
    table.add_column("Score", width=8)
    table.add_column("File", overflow="fold")
    table.add_column("Reasons", overflow="fold")

    for _, row in flagged.head(top_n).iterrows():
        label, colour = severity_label(row)
        severity_text = Text(label, style=f"bold {colour}")

        table.add_row(
            severity_text,
            f"{row['entropy']:.2f}",
            f"{row['anomaly_score']:.4f}",
            str(row["filepath"]),
            explain_file(row),
        )

    console.print(table)
    console.print()

    # Summary breakdown
    high = sum(1 for _, r in flagged.iterrows() if severity_label(r["anomaly_score"])[0] == "HIGH")
    medium = sum(1 for _, r in flagged.iterrows() if severity_label(r["anomaly_score"])[0] == "MEDIUM")
    low = sum(1 for _, r in flagged.iterrows() if severity_label(r["anomaly_score"])[0] == "LOW")

    console.print(f"  [red]HIGH[/red]   {high}")
    console.print(f"  [yellow]MEDIUM[/yellow] {medium}")
    console.print(f"  [cyan]LOW[/cyan]    {low}")
    console.print()


def export_csv(results: pd.DataFrame, output_path: str):
    """Export full results (flagged and clean) to CSV for further analysis."""
    results.to_csv(output_path, index=False)
    console.print(f"[dim]Results exported to {output_path}[/dim]")