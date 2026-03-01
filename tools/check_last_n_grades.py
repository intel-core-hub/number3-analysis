import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (apply_grade_veto, compute_composite_score, compute_grade,
                 compute_risk_adjusted_ev)

CSV = ROOT / "artifacts" / "historical_backtest_1y.csv"


def main(n: int = 100) -> None:
    if not CSV.exists():
        print(f"Missing file: {CSV}")
        return

    df = pd.read_csv(CSV)
    if df.empty:
        print("No rows in historical CSV")
        return

    df = df.tail(n).copy()

    ev = df.get("ev_ratio")
    if ev is None:
        df["ev_ratio"] = 0.0
    else:
        df["ev_ratio"] = pd.to_numeric(ev, errors="coerce").fillna(0.0)

    conf = df.get("confidence")
    if conf is None:
        df["confidence"] = 0.0
    else:
        df["confidence"] = pd.to_numeric(conf, errors="coerce").fillna(0.0)

    # historical data doesn't include ruin_probability; set to 0.0 for this check
    df["ruin_probability"] = 0.0

    def _radv_row(r: pd.Series) -> float:
        return compute_risk_adjusted_ev(
            float(r.get("ev_ratio", 0.0)), float(r.get("confidence", 0.0))
        )

    df["risk_adjusted_ev"] = df.apply(_radv_row, axis=1)

    def _composite_row(r: pd.Series) -> float:
        return compute_composite_score(
            float(r.get("ev_ratio", 0.0)),
            float(r.get("confidence", 0.0)),
            float(r.get("ruin_probability", 0.0)),
            float(r.get("risk_adjusted_ev", 0.0)),
        )

    df["composite_score"] = df.apply(_composite_row, axis=1)

    df["grade_before_veto"] = df["composite_score"].apply(lambda s: compute_grade(s))

    def _grade_veto_row(r: pd.Series) -> str:
        return apply_grade_veto(
            str(r.get("grade_before_veto", "E")),
            float(r.get("risk_adjusted_ev", 0.0)),
            float(r.get("ruin_probability", 0.0)),
            float(r.get("confidence", 0.0)),
            float(r.get("ev_ratio", 0.0)),
        )

    df["grade"] = df.apply(_grade_veto_row, axis=1)

    counts = df["grade"].value_counts().to_dict()
    print(f"Checked last {n} rows. Grade counts (after veto):")
    for g in ["S", "A", "B", "C", "D", "E"]:
        print(f"  {g}: {counts.get(g, 0)}")

    positive = df[df["grade"].isin(["D", "C", "B", "A", "S"])].copy()
    if positive.empty:
        print(f"No rows with grade D or higher in last {n} rows.")
    else:
        print(
            f"Found {len(positive)} rows with grade >= D in last {n} rows. Showing them:"
        )
        out = positive.sort_values(
            ["composite_score", "ev_ratio"], ascending=[False, False]
        )
        cols = [
            "draw_date",
            "round_idx",
            "recommended",
            "ev_ratio",
            "confidence",
            "risk_adjusted_ev",
            "composite_score",
            "grade",
        ]
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=100, help="Number of last rows to check")
    args = p.parse_args()
    main(args.n)
