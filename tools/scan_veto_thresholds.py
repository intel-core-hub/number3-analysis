import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (calibrate_confidence, compute_composite_score, compute_grade,
                 compute_risk_adjusted_ev)

CSV = ROOT / "artifacts" / "historical_backtest_1y.csv"


def main() -> None:
    if not CSV.exists():
        print(f"Missing file: {CSV}")
        return
    df = pd.read_csv(CSV)
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
    df["ruin_probability"] = 0.0

    import numpy as np

    veto_thresholds = [round(x, 4) for x in list(np.linspace(0.001, 0.01, 19))]

    rows = []
    for vt in veto_thresholds:
        # apply veto by setting grade to E when confidence < vt
        tmp = df.copy()

        def _calib(c: float) -> float:
            return calibrate_confidence(c)

        tmp["confidence_calibrated"] = tmp["confidence"].apply(_calib)

        def _radv(r: pd.Series) -> float:
            return compute_risk_adjusted_ev(
                float(r.get("ev_ratio", 0.0)), float(r.get("confidence", 0.0))
            )

        tmp["risk_adjusted_ev"] = tmp.apply(_radv, axis=1)

        def _comp(r: pd.Series) -> float:
            return compute_composite_score(
                float(r.get("ev_ratio", 0.0)),
                float(r.get("confidence", 0.0)),
                float(r.get("ruin_probability", 0.0)),
                float(r.get("risk_adjusted_ev", 0.0)),
            )

        tmp["composite_score"] = tmp.apply(_comp, axis=1)

        tmp["grade_base"] = tmp["composite_score"].apply(lambda s: compute_grade(s))

        def _grade_veto(r: pd.Series) -> str:
            conf_val = float(r.get("confidence_calibrated", 0.0))
            radv_val = float(r.get("risk_adjusted_ev", 0.0))
            return (
                "E"
                if (conf_val < vt or radv_val < 0)
                else str(r.get("grade_base", "E"))
            )

        tmp["grade_vetoed"] = tmp.apply(_grade_veto, axis=1)

        vc = tmp["grade_vetoed"].value_counts().to_dict()
        total_profit = tmp[tmp["grade_vetoed"] != "E"]["profit"].sum()
        total_cost = tmp[tmp["grade_vetoed"] != "E"]["total_cost"].sum()
        roi = (total_profit / total_cost) if total_cost > 0 else 0.0

        rows.append(
            {
                "veto_threshold": vt,
                "count_notE": int((tmp["grade_vetoed"] != "E").sum()),
                "profit": float(total_profit),
                "cost": float(total_cost),
                "roi": roi,
                "vc": vc,
            }
        )

    out = ROOT / "artifacts" / "veto_threshold_scan.csv"
    import csv

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["veto_threshold", "count_notE", "profit", "cost", "roi", "grades_json"]
        )
        for r in rows:
            w.writerow(
                [
                    f"{r['veto_threshold']:.4f}",
                    r["count_notE"],
                    f"{r['profit']:.2f}",
                    f"{r['cost']:.2f}",
                    f"{r['roi']:.6f}",
                    str(r["vc"]),
                ]
            )

    print(f"Wrote veto threshold scan to: {out}")


if __name__ == "__main__":
    main()
