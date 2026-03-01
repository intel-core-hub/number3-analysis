import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (apply_grade_veto, compute_composite_score, compute_grade,
                 compute_risk_adjusted_ev)

CSV = ROOT / "artifacts" / "historical_backtest_1y.csv"


def analyze(df: pd.DataFrame, use_veto: bool = True) -> pd.DataFrame:
    # compute radv and score
    df = df.copy()

    def _radv_row(r: pd.Series) -> float:
        return compute_risk_adjusted_ev(
            float(r.get("ev_ratio", 0.0) or 0.0), float(r.get("confidence", 0.0) or 0.0)
        )

    df["risk_adjusted_ev"] = df.apply(_radv_row, axis=1)

    def _composite_row(r: pd.Series) -> float:
        return compute_composite_score(
            float(r.get("ev_ratio", 0.0) or 0.0),
            float(r.get("confidence", 0.0) or 0.0),
            float(r.get("ruin_probability", 0.0) or 0.0),
            float(r.get("risk_adjusted_ev", 0.0) or 0.0),
        )

    df["composite_score"] = df.apply(_composite_row, axis=1)

    df["grade_base"] = df["composite_score"].apply(lambda s: compute_grade(s))
    if use_veto:

        def _grade_row(r: pd.Series) -> str:
            return apply_grade_veto(
                r["grade_base"],
                float(r.get("risk_adjusted_ev", 0.0) or 0.0),
                float(r.get("ruin_probability", 0.0) or 0.0),
                float(r.get("confidence", 0.0) or 0.0),
                float(r.get("ev_ratio", 0.0) or 0.0),
            )

        df["grade"] = df.apply(_grade_row, axis=1)
    else:
        df["grade"] = df["grade_base"].copy()
    return df


def main() -> None:
    if not CSV.exists():
        print(f"Missing file: {CSV}")
        return

    df = pd.read_csv(CSV)
    if df.empty:
        print("No rows in historical CSV")
        return

    # Ensure numeric
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

    total_cost = df.get("total_cost")
    if total_cost is None:
        df["total_cost"] = 0.0
    else:
        df["total_cost"] = pd.to_numeric(total_cost, errors="coerce").fillna(0.0)

    profit = df.get("profit")
    if profit is None:
        df["profit"] = 0.0
    else:
        df["profit"] = pd.to_numeric(profit, errors="coerce").fillna(0.0)

    # no ruin in historical; assume 0
    df["ruin_probability"] = 0.0

    thresholds = [i / 10000.0 for i in range(0, 101)]  # 0.0000 .. 0.0100 step 0.0001

    results: list[dict[str, Any]] = []
    for t in thresholds:
        subset = df[df["confidence"] >= t].copy()
        count = len(subset)
        avg_ev = float(subset["ev_ratio"].mean()) if count > 0 else 0.0
        sum_profit = float(subset["profit"].sum()) if count > 0 else 0.0
        sum_cost = float(subset["total_cost"].sum()) if count > 0 else 0.0
        roi = (sum_profit / sum_cost) if sum_cost > 0 else 0.0

        analyzed_no_veto = analyze(subset, use_veto=False) if count > 0 else subset
        analyzed_veto = analyze(subset, use_veto=True) if count > 0 else subset

        # grade distributions
        def grade_counts(df_in: pd.DataFrame | None) -> dict[str, int]:
            if df_in is None or df_in.empty:
                return {g: 0 for g in ["S", "A", "B", "C", "D", "E"]}
            vc = df_in["grade"].value_counts().to_dict()
            return {g: int(vc.get(g, 0)) for g in ["S", "A", "B", "C", "D", "E"]}

        counts_no_veto = grade_counts(analyzed_no_veto)
        counts_veto = grade_counts(analyzed_veto)

        avg_score = (
            float(analyzed_no_veto["composite_score"].mean()) if count > 0 else 0.0
        )

        results.append(
            {
                "threshold": round(t, 4),
                "count": count,
                "avg_ev": round(avg_ev, 4),
                "sum_profit": round(sum_profit, 2),
                "sum_cost": round(sum_cost, 2),
                "roi": round(roi, 6),
                "avg_score": round(avg_score, 2),
                "counts_no_veto": counts_no_veto,
                "counts_veto": counts_veto,
            }
        )

    # print summary header
    # write CSV to artifacts for further analysis
    out_csv = ROOT / "artifacts" / "sensitivity_confidence_scan.csv"
    import csv

    header = [
        "threshold",
        "count",
        "avg_ev",
        "sum_profit",
        "sum_cost",
        "roi",
        "avg_score",
        "S_no",
        "A_no",
        "B_no",
        "C_no",
        "D_no",
        "E_no",
        "S_v",
        "A_v",
        "B_v",
        "C_v",
        "D_v",
        "E_v",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for rr in results:
            counts_no = cast(dict[str, int], rr["counts_no_veto"])
            counts_v = cast(dict[str, int], rr["counts_veto"])
            row = [
                f"{rr['threshold']:.4f}",
                rr["count"],
                f"{rr['avg_ev']:.4f}",
                f"{rr['sum_profit']:.2f}",
                f"{rr['sum_cost']:.2f}",
                f"{rr['roi']:.6f}",
                f"{rr['avg_score']:.2f}",
                counts_no["S"],
                counts_no["A"],
                counts_no["B"],
                counts_no["C"],
                counts_no["D"],
                counts_no["E"],
                counts_v["S"],
                counts_v["A"],
                counts_v["B"],
                counts_v["C"],
                counts_v["D"],
                counts_v["E"],
            ]
            w.writerow(row)

    print(f"Wrote results to: {out_csv}")
    # quick summary: thresholds that produced any D+ without veto and any with positive ROI
    df_res = pd.read_csv(out_csv)
    any_dpos = df_res[
        df_res["D_no"]
        + df_res["C_no"]
        + df_res["B_no"]
        + df_res["A_no"]
        + df_res["S_no"]
        > 0
    ]
    pos_roi = df_res[df_res["roi"] > 0]
    print(f"Thresholds with any grade >=D (no veto): {len(any_dpos)}")
    print(f"Thresholds with positive ROI: {len(pos_roi)}")
    if not any_dpos.empty:
        print("Example thresholds with D+ (no veto):")
        print(any_dpos.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
