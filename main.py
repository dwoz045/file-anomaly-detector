import argparse
import sys
import pandas as pd
from scanner import scan_directory
from model import train_model, score_files, save_model, load_model
from report import print_report, export_csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="File Anomaly Detector — identify suspicious files using ML"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- scan command ---
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan a directory and report anomalies"
    )
    scan_parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory to scan"
    )
    scan_parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="Expected proportion of anomalous files (default: 0.01)"
    )
    scan_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of anomalous files to display (default: 20)"
    )
    scan_parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Optional path to export full results as CSV"
    )
    scan_parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model to disk for reuse"
    )

    # --- rescan command ---
    rescan_parser = subparsers.add_parser(
        "rescan",
        help="Scan a directory using a previously saved model"
    )
    rescan_parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory to scan"
    )
    rescan_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of anomalous files to display (default: 20)"
    )
    rescan_parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Optional path to export full results as CSV"
    )

    # --- monitor command ---
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Continuously scan a directory and report changes between scans"
    )
    monitor_parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory to monitor"
    )
    monitor_parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between scans (default: 60)"
    )
    monitor_parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Stop after this many scans (default: run forever)"
    )

    return parser.parse_args()


def run_scan(directory: str, contamination: float, top: int, export: str, save: bool):
    if not Path(directory).exists():
        print(f"[error] Directory not found: {directory}")
        sys.exit(1)

    df = scan_directory(directory)

    if len(df) < 10:
        print("[error] Too few files to build a meaningful model. Try a larger directory.")
        sys.exit(1)

    model, scaler, clean_df = train_model(df, contamination=contamination)
    results = score_files(clean_df, model, scaler)
    print_report(results, top_n=top)

    if save:
        save_model(model, scaler)

    if export:
        export_csv(results, export)


def run_rescan(directory: str, top: int, export: str):
    if not Path(directory).exists():
        print(f"[error] Directory not found: {directory}")
        sys.exit(1)

    try:
        model, scaler = load_model()
    except FileNotFoundError:
        print("[error] No saved model found. Run 'scan' with --save-model first.")
        sys.exit(1)

    df = scan_directory(directory)
    results = score_files(df, model, scaler)
    print_report(results, top_n=top)

    if export:
        export_csv(results, export)


def run_monitor_command(directory: str, interval: int, max_scans: int):
    from monitor import run_monitor, add_file_hashes
    from autoencoder import train_autoencoder, score_files_autoencoder, combined_anomaly_score
    from yara_scanner import compile_rules, scan_dataframe

    if not Path(directory).exists():
        print(f"[error] Directory not found: {directory}")
        sys.exit(1)

    rules = compile_rules()

    def full_scan(target_dir: str) -> pd.DataFrame:
        df = scan_directory(target_dir)
        model, scaler, clean_df = train_model(df)
        results = score_files(clean_df, model, scaler)
        ae_model, ae_scaler, threshold = train_autoencoder(clean_df)
        results = score_files_autoencoder(results, ae_model, ae_scaler, threshold)
        results = combined_anomaly_score(results)
        results = scan_dataframe(results, rules)
        return results

    run_monitor(directory, full_scan, interval_seconds=interval, max_scans=max_scans)


def main():
    args = parse_args()

    if args.command == "scan":
        run_scan(
            directory=args.directory,
            contamination=args.contamination,
            top=args.top,
            export=args.export,
            save=args.save_model
        )

    elif args.command == "rescan":
        run_rescan(
            directory=args.directory,
            top=args.top,
            export=args.export
        )

    elif args.command == "monitor":
        run_monitor_command(args.directory, args.interval, args.max_scans)


if __name__ == "__main__":
    main()