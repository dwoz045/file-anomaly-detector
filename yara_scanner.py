import yara
import pathlib
import pandas as pd
from rich.console import Console


console = Console()

RULES_DIR = pathlib.Path(__file__).parent / "data" / "yara_rules"


def compile_rules(rules_dir: str = None) -> yara.Rules:
    """
    Compile all .yar/.yara files in the rules directory into
    a single YARA ruleset.
    """
    rules_path = pathlib.Path(rules_dir) if rules_dir else RULES_DIR

    rule_files = {}
    for i, path in enumerate(rules_path.glob("*.yar*")):
        rule_files[f"rule_{i}"] = str(path)

    if not rule_files:
        raise FileNotFoundError(f"No YARA rule files found in {rules_path}")

    print(f"[yara] Compiling {len(rule_files)} rule file(s)...")
    rules = yara.compile(filepaths=rule_files)
    print("[yara] Rules compiled successfully.")
    return rules


def scan_file(filepath: str, rules: yara.Rules, timeout: int = 5) -> list[dict]:
    """
    Run YARA rules against a single file.
    Returns a list of match dicts, empty if no matches.
    Each dict contains rule name, tags, and matched strings.
    """
    try:
        matches = rules.match(filepath, timeout=timeout)
        return [
            {
                "rule": match.rule,
                "tags": match.tags,
                "meta": match.meta,
                "strings": [
                    {
                        "identifier": s.identifier,
                        "offset": s.instances[0].offset if s.instances else None
                    }
                    for s in match.strings
                ]
            }
            for match in matches
        ]
    except yara.TimeoutError:
        return [{"rule": "TIMEOUT", "tags": [], "meta": {}, "strings": []}]
    except (PermissionError, OSError, yara.Error):
        return []


def scan_dataframe(
    df: pd.DataFrame,
    rules: yara.Rules,
    flagged_only: bool = True
) -> pd.DataFrame:
    """
    Run YARA against files in a DataFrame.

    If flagged_only=True, only scans files already flagged as anomalous
    by the ML models — much faster for large scans.
    If flagged_only=False, scans every file (slow on large directories).

    Adds columns:
      - yara_matches: number of rules matched
      - yara_rules_hit: comma-separated rule names
      - yara_hit: 1 if any rule matched, 0 otherwise
    """
    df = df.copy()

    if flagged_only and "consensus" in df.columns:
        scan_mask = df["consensus"].isin(["both_flagged", "if_only", "ae_only"])
        targets = df[scan_mask].copy()
        print(f"[yara] Scanning {len(targets)} flagged files (flagged_only=True)...")
    else:
        targets = df.copy()
        print(f"[yara] Scanning all {len(targets)} files...")

    match_counts = []
    rule_names = []
    hit_flags = []

    for _, row in targets.iterrows():
        matches = scan_file(row["filepath"], rules)
        real_matches = [m for m in matches if m["rule"] != "TIMEOUT"]
        match_counts.append(len(real_matches))
        rule_names.append(", ".join(m["rule"] for m in real_matches) if real_matches else "")
        hit_flags.append(1 if real_matches else 0)

    targets["yara_matches"] = match_counts
    targets["yara_rules_hit"] = rule_names
    targets["yara_hit"] = hit_flags

    # Merge back into full dataframe
    df = df.merge(
        targets[["filepath", "yara_matches", "yara_rules_hit", "yara_hit"]],
        on="filepath",
        how="left"
    )
    df["yara_matches"] = df["yara_matches"].fillna(0).astype(int)
    df["yara_rules_hit"] = df["yara_rules_hit"].fillna("")
    df["yara_hit"] = df["yara_hit"].fillna(0).astype(int)

    return df


def threat_level(row: pd.Series) -> str:
    """
    Compute a final threat level combining ML consensus and YARA signal.

    CRITICAL  — flagged by both ML models AND matched a YARA rule
    HIGH      — flagged by both ML models
    MEDIUM    — flagged by one ML model or matched a YARA rule
    LOW       — minor statistical anomaly only
    """
    yara_hit = row.get("yara_hit", 0) == 1
    consensus = row.get("consensus", "clean")

    if consensus == "both_flagged" and yara_hit:
        return "CRITICAL"
    elif consensus == "both_flagged":
        return "HIGH"
    elif consensus in ("if_only", "ae_only") or yara_hit:
        return "MEDIUM"
    else:
        return "LOW"


def print_yara_summary(df: pd.DataFrame):
    """Print a summary of YARA hits to the console."""
    hits = df[df["yara_hit"] == 1]

    if hits.empty:
        console.print("[green][yara] No YARA rule matches found.[/green]")
        return

    console.print(f"\n[bold red][yara] {len(hits)} file(s) matched YARA rules:[/bold red]")
    for _, row in hits.iterrows():
        console.print(f"  [red]•[/red] {row['filepath']}")
        console.print(f"    [dim]Rules: {row['yara_rules_hit']}[/dim]")