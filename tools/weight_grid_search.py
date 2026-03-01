import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import compute_grade, compute_risk_adjusted_ev

CSV = ROOT / "artifacts" / "historical_backtest_1y.csv"


def compute_with_weights(
    df: pd.DataFrame, ev_w: float, conf_w: float, radv_w: float, ruin_w: float
) -> pd.DataFrame:
    # temporarily override global weights by passing them into compute_composite_score via monkeypatch
    # simpler approach: re-implement normalization inline for the grid search
    df = df.copy()

    def _radv_row(r: pd.Series) -> float:
        return compute_risk_adjusted_ev(
            float(r.get("ev_ratio", 0.0) or 0.0), float(r.get("confidence", 0.0) or 0.0)
        )

    df["risk_adjusted_ev"] = df.apply(_radv_row, axis=1)

    def score_row(ev: float, conf: float, ruin: float, radv: float) -> float:
        def _clamp01(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        ev_norm = _clamp01((ev - 1.00) / 0.20)
        conf_norm = _clamp01(conf)
        radv_norm = _clamp01((radv + 0.20) / 0.30)
        ruin_norm = _clamp01(1.0 - (ruin / 0.30))
        score = 100.0 * (
            ev_w * ev_norm
            + conf_w * conf_norm
            + radv_w * radv_norm
            + ruin_w * ruin_norm
        )
        return round(score, 2)

    def _score_row_apply(r: pd.Series) -> float:
        return score_row(
            float(r["ev_ratio"]),
            float(r["confidence"]),
            float(r.get("ruin_probability", 0.0)),
            float(r["risk_adjusted_ev"]),
        )

    df["composite_score"] = df.apply(_score_row_apply, axis=1)
    df["grade"] = df["composite_score"].apply(lambda s: compute_grade(s))
    return df


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

    profit = df.get("profit")
    if profit is None:
        df["profit"] = 0.0
    else:
        df["profit"] = pd.to_numeric(profit, errors="coerce").fillna(0.0)

    total_cost = df.get("total_cost")
    if total_cost is None:
        df["total_cost"] = 0.0
    else:
        df["total_cost"] = pd.to_numeric(total_cost, errors="coerce").fillna(0.0)
    df["ruin_probability"] = 0.0

    # grid: ev_w in [0.2,0.3,0.4,0.5], conf_w in [0.1,0.2,0.3,0.4]
    results = []
    for ev_w in [0.2, 0.3, 0.4, 0.5]:
        for conf_w in [0.1, 0.2, 0.3, 0.4]:
            radv_w = 0.25
            ruin_w = 1.0 - (ev_w + conf_w + radv_w)
            if ruin_w < 0:
                continue
            df_sc = compute_with_weights(df, ev_w, conf_w, radv_w, ruin_w)
            # summarize grades (no veto)
            vc = df_sc["grade"].value_counts().to_dict()
            total_profit = df_sc[df_sc["grade"] != "E"]["profit"].sum()
            total_cost = df_sc[df_sc["grade"] != "E"]["total_cost"].sum()
            roi = (total_profit / total_cost) if total_cost > 0 else 0.0
            results.append(
                {
                    "ev_w": ev_w,
                    "conf_w": conf_w,
                    "radv_w": radv_w,
                    "ruin_w": ruin_w,
                    "counts": vc,
                    "roi": roi,
                }
            )

    out = ROOT / "artifacts" / "weight_grid_search.csv"
    import csv

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ev_w", "conf_w", "radv_w", "ruin_w", "roi", "counts"])
        for r in results:
            w.writerow(
                [
                    r["ev_w"],
                    r["conf_w"],
                    r["radv_w"],
                    r["ruin_w"],
                    f"{r['roi']:.6f}",
                    str(r["counts"]),
                ]
            )
    print(f"Wrote weight grid results to: {out}")


if __name__ == "__main__":
    main()
