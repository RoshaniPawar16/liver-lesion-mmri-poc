#!/usr/bin/env python3
"""Replace <!-- RESULT: name --> placeholders in README.md from results_summary.csv.

Each placeholder is matched to a column or derived value in results_summary.csv.

Usage:
    python src/fill_readme.py \
        --summary reports/results_summary.csv \
        --readme README.md
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def build_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Return {placeholder_name: formatted_string} for every result."""
    lookup: dict[str, str] = {}

    for _, row in df.iterrows():
        run = row["run"]

        def fmt(val: float, decimals: int = 3) -> str:
            return f"{val:.{decimals}f}"

        def ci(col: str) -> str:
            return f"{fmt(row[col])} [{fmt(row[col+'_lo'])}-{fmt(row[col+'_hi'])}]"

        lookup[f"{run}_auroc"] = ci("auroc")
        lookup[f"{run}_auprc"] = ci("auprc")
        lookup[f"{run}_sens"] = fmt(row["sensitivity"])
        lookup[f"{run}_spec"] = fmt(row["specificity"])
        lookup[f"{run}_ece"] = fmt(row["ece"])

    # Best model ECE (first row, sorted descending by AUROC)
    if not df.empty:
        lookup["best_ece"] = fmt(df.iloc[0]["ece"])

    return lookup


def fill(text: str, lookup: dict[str, str]) -> tuple[str, list[str]]:
    """Replace all placeholders. Return (filled_text, list_of_unfilled)."""
    unfilled: list[str] = []

    def replace(m: re.Match) -> str:
        key = m.group(1)
        if key in lookup:
            return lookup[key]
        unfilled.append(key)
        return m.group(0)  # leave unchanged

    filled = re.sub(r"<!-- RESULT: ([^>]+) -->", replace, text)
    return filled, unfilled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="reports/results_summary.csv")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    lookup = build_lookup(df)

    readme_path = Path(args.readme)
    text = readme_path.read_text()
    filled, unfilled = fill(text, lookup)

    readme_path.write_text(filled)
    print(f"Filled {len(lookup) - len(unfilled)} placeholders in {readme_path}")
    if unfilled:
        print(f"Unfilled ({len(unfilled)}): {unfilled}")


if __name__ == "__main__":
    main()
