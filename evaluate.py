import pandas as pd
import numpy as np
import pathlib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from rich.console import Console
from rich.table import Table
from rich import box

from scanner import scan_directory
from model import train_model, score_files
from autoencoder import train_autoencoder, score_files_autoencoder, combined_anomaly_score
from yara_scanner import compile_rules, scan_dataframe


console = Console()

TEST_DIR     = pathlib.Path(__file__).parent / "data" / "test_env"
CLEAN_DIR    = TEST_DIR / "clean"
MALICIOUS_DIR = TEST_DIR / "malicious"


# ─── Ground Truth ─────────────────────────────────────────────────────────────

def build_labelled_dataset() -> pd.DataFrame:
    """
    Scan clean and malicious directories separately,
    then combine into a single labelled DataFrame.
    label=0 means clean, label=1 means malicious.
    """
    console.print("[evaluate] Scanning clean files...")
    clean_df = scan_directory(str(CLEAN_DIR))
    clean_df["label"] = 0

    console.print("[evaluate] Scanning malicious files...")
    mal_df = scan_directory(str(MALICIOUS_DIR))
    mal_df["label"] = 1

    combined = pd.concat([clean_df, mal_df], ignore_index=True)
    console.print(
        f"[evaluate] Dataset: {len(clean_df)} clean + "
        f"{len(mal_df)} malicious = {len(combined)} total files"
    )
    return combined


# ─── Model Evaluation ─────────────────────────────────────────────────────────

def evaluate_isolation_forest(df: pd.DataFrame) -> dict:
    """
    Train Isolation Forest on the full dataset and evaluate
    against ground truth labels.
    """
    console.print("\n[evaluate] Evaluating Isolation Forest...")

    # Train on clean files only — this is realistic:
    # in production you'd train on known-good data
    clean_df = df[df["label"] == 0].copy()
    model, scaler, clean_prepared = train_model(clean_df, contamination=0.05)

    # Score everything
    results = score_files(df, model, scaler)
    results["label"] = df["label"].values

    y_true = results["label"]
    y_pred = results["is_anomaly"]

    return {
        "model": "Isolation Forest",
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, -results["anomaly_score"]),
        "confusion": confusion_matrix(y_true, y_pred),
        "results_df": results,
    }


def evaluate_autoencoder(df: pd.DataFrame) -> dict:
    """
    Train autoencoder on clean files only and evaluate
    against ground truth labels.
    """
    console.print("[evaluate] Evaluating Autoencoder...")

    clean_df = df[df["label"] == 0].copy()
    ae_model, ae_scaler, threshold = train_autoencoder(
        clean_df,
        epochs=150,
        threshold_percentile=95.0
    )

    results = score_files_autoencoder(df, ae_model, ae_scaler, threshold)
    results["label"] = df["label"].values

    y_true = results["label"]
    y_pred = results["ae_is_anomaly"]

    return {
        "model": "Autoencoder",
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, results["ae_reconstruction_error"]),
        "confusion": confusion_matrix(y_true, y_pred),
        "results_df": results,
    }


def evaluate_yara(df: pd.DataFrame) -> dict:
    """
    Run YARA rules against all files and evaluate against ground truth.
    """
    console.print("[evaluate] Evaluating YARA rules...")

    rules = compile_rules()
    results = scan_dataframe(df, rules, flagged_only=False)
    results["label"] = df["label"].values

    y_true = results["label"]
    y_pred = results["yara_hit"]

    return {
        "model": "YARA Rules",
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, y_pred),
        "confusion": confusion_matrix(y_true, y_pred),
        "results_df": results,
    }


def evaluate_ensemble(df: pd.DataFrame, if_results: dict, ae_results: dict) -> dict:
    """
    Evaluate the combined ensemble — flagged by both IF and AE.
    This is the highest confidence signal.
    """
    console.print("[evaluate] Evaluating Ensemble (IF + AE consensus)...")

    if_df = if_results["results_df"][["filepath", "is_anomaly", "anomaly_score"]].copy()
    ae_df = ae_results["results_df"][["filepath", "ae_is_anomaly", "ae_reconstruction_error"]].copy()

    merged = if_df.merge(ae_df, on="filepath")
    merged["label"] = df["label"].values
    merged = combined_anomaly_score(merged)

    # Ensemble prediction: only flag if both models agree
    merged["ensemble_pred"] = (
        (merged["is_anomaly"] == 1) & (merged["ae_is_anomaly"] == 1)
    ).astype(int)

    y_true = merged["label"]
    y_pred = merged["ensemble_pred"]

    return {
        "model": "Ensemble (IF + AE)",
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, -merged["anomaly_score"] + merged["ae_reconstruction_error"]),
        "confusion": confusion_matrix(y_true, y_pred),
        "results_df": merged,
    }


# ─── False Positive Rate ──────────────────────────────────────────────────────

def false_positive_rate(confusion: np.ndarray) -> float:
    """
    Compute false positive rate from a confusion matrix.
    FPR = FP / (FP + TN)
    """
    tn, fp, fn, tp = confusion.ravel()
    if (fp + tn) == 0:
        return 0.0
    return fp / (fp + tn)


# ─── Report ───────────────────────────────────────────────────────────────────

def print_evaluation_report(results: list[dict]):
    """
    Print a rich formatted comparison table of all model evaluations.
    These are the numbers you put on your CV.
    """
    console.print()
    console.print("[bold white]EVALUATION REPORT[/bold white]", justify="center")
    console.print("[dim]Trained on clean files only — tested on clean + malicious[/dim]", justify="center")
    console.print()

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold dim")
    table.add_column("Model",      width=22)
    table.add_column("Precision",  width=10)
    table.add_column("Recall",     width=10)
    table.add_column("F1",         width=10)
    table.add_column("AUC-ROC",    width=10)
    table.add_column("FPR",        width=10)

    for r in results:
        fpr = false_positive_rate(r["confusion"])
        table.add_row(
            r["model"],
            f"{r['precision']:.1%}",
            f"{r['recall']:.1%}",
            f"{r['f1']:.1%}",
            f"{r['auc_roc']:.3f}",
            f"{fpr:.1%}",
        )

    console.print(table)

    # Per-file breakdown
    console.print("\n[bold dim]Per-file breakdown (malicious files):[/bold dim]")
    for r in results:
        df = r["results_df"]
        if "label" not in df.columns:
            continue

        mal = df[df["label"] == 1]
        pred_col = (
            "ensemble_pred" if "ensemble_pred" in df.columns
            else "yara_hit" if "yara_hit" in df.columns
            else "ae_is_anomaly" if "ae_is_anomaly" in df.columns
            else "is_anomaly"
        )

        console.print(f"\n  [dim]{r['model']}[/dim]")
        for _, row in mal.iterrows():
            name = pathlib.Path(row["filepath"]).name
            caught = row.get(pred_col, 0) == 1
            status = "[green]CAUGHT[/green]" if caught else "[red]MISSED[/red]"
            console.print(f"    {status}  {name}")

    console.print()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def run_evaluation():
    console.print("\n[bold green]═══ FILE ANOMALY DETECTOR — EVALUATION ═══[/bold green]\n")

    df = build_labelled_dataset()

    if_result  = evaluate_isolation_forest(df)
    ae_result  = evaluate_autoencoder(df)
    yara_result = evaluate_yara(df)
    ens_result = evaluate_ensemble(df, if_result, ae_result)

    print_evaluation_report([if_result, ae_result, yara_result, ens_result])

def build_large_labelled_dataset() -> pd.DataFrame:
    clean_dir     = TEST_DIR / "clean_large"
    malicious_dir = TEST_DIR / "malicious_large"

    console.print("[evaluate] Scanning large clean dataset...")
    clean_df = scan_directory(str(clean_dir))
    clean_df["label"] = 0
    clean_df["family"] = "clean"

    console.print("[evaluate] Scanning malicious samples by family...")
    family_dfs = []

    for family_dir in sorted(malicious_dir.iterdir()):
        if not family_dir.is_dir():
            continue
        fdf = scan_directory(str(family_dir))
        if fdf.empty:
            continue
        fdf["label"] = 1
        fdf["family"] = family_dir.name
        family_dfs.append(fdf)
        console.print(f"  [dim]{family_dir.name}: {len(fdf)} samples[/dim]")

    mal_df = pd.concat(family_dfs, ignore_index=True)
    combined = pd.concat([clean_df, mal_df], ignore_index=True)

    console.print(
        f"[evaluate] Dataset: {len(clean_df)} clean + "
        f"{len(mal_df)} malicious = {len(combined)} total"
    )
    return combined


def per_family_breakdown(results_df: pd.DataFrame, pred_col: str, model_name: str):
    if "family" not in results_df.columns:
        return

    console.print(f"\n[bold dim]Detection rate by family — {model_name}:[/bold dim]")

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Family",         width=30)
    table.add_column("Samples",        width=10)
    table.add_column("Detected",       width=10)
    table.add_column("Detection Rate", width=16)

    families = results_df[results_df["family"] != "clean"]["family"].unique()

    for family in sorted(families):
        fdf      = results_df[results_df["family"] == family]
        total    = len(fdf)
        detected = int(fdf[pred_col].sum())
        rate     = detected / total if total > 0 else 0.0
        colour   = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
        table.add_row(family, str(total), str(detected), f"[{colour}]{rate:.1%}[/{colour}]")

    console.print(table)


def run_large_evaluation():
    console.print("\n[bold green]═══ LARGE SCALE EVALUATION ═══[/bold green]\n")

    df = build_large_labelled_dataset()

    if_result   = evaluate_isolation_forest(df)
    ae_result   = evaluate_autoencoder(df)
    yara_result = evaluate_yara(df)
    ens_result  = evaluate_ensemble(df, if_result, ae_result)
    print_evaluation_report([if_result, ae_result, yara_result, ens_result])

    yara_df          = yara_result["results_df"].copy()
    yara_df["family"] = df["family"].values
    per_family_breakdown(yara_df, "yara_hit", "YARA")

    ae_df            = ae_result["results_df"].copy()
    ae_df["family"]   = df["family"].values
    per_family_breakdown(ae_df, "ae_is_anomaly", "Autoencoder")


if __name__ == "__main__":
    run_large_evaluation()