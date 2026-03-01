from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (EV_THRESHOLD_DEFAULT, apply_grade_veto, calibrate_confidence,
                 compute_composite_score, compute_grade,
                 compute_risk_adjusted_ev, estimate_hit_probability,
                 probability_edge, required_hit_probability)

INPUT_CSV = ROOT / "artifacts" / "historical_backtest_1y.csv"
META_THRESHOLD_JSON = ROOT / "artifacts" / "meta_threshold.json"
OUT_SUMMARY = ROOT / "artifacts" / "e2e_revalidation_summary.csv"
OUT_DETAIL = ROOT / "artifacts" / "e2e_revalidation_detail.csv"


def _load_meta_threshold(default: float = 0.65) -> float:
    if not META_THRESHOLD_JSON.exists():
        return default
    try:
        payload = json.loads(META_THRESHOLD_JSON.read_text(encoding="utf-8"))
        value = float(payload.get("best_threshold", default))
    except Exception:
        return default
    return max(0.5, min(0.95, value))


def _pick_cost_column(df: pd.DataFrame) -> pd.Series:
    if "total_cost" in df.columns:
        return pd.to_numeric(df["total_cost"], errors="coerce").fillna(0.0)
    if "cost" in df.columns:
        return pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)
    return pd.Series(200.0, index=df.index, dtype=float)


def _pick_profit_column(df: pd.DataFrame) -> pd.Series:
    if "profit" in df.columns:
        return pd.to_numeric(df["profit"], errors="coerce").fillna(0.0)
    if "set_profit" in df.columns:
        return pd.to_numeric(df["set_profit"], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _get_numeric_column(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _summarize(name: str, frame: pd.DataFrame, buy_col: str) -> dict[str, float | str]:
    bought = frame[frame[buy_col]].copy()
    bets = int(len(bought))
    cost = float(bought["cost_if_buy"].sum()) if bets > 0 else 0.0
    profit = float(bought["profit_if_buy"].sum()) if bets > 0 else 0.0
    roi = (profit / cost) if cost > 0 else 0.0
    hit_rate = float((bought["profit_if_buy"] > 0).mean()) if bets > 0 else 0.0
    avg_ev = float(bought["ev_ratio"].mean()) if bets > 0 else 0.0
    avg_conf = float(bought["confidence"].mean()) if bets > 0 else 0.0
    avg_score = float(bought["composite_score"].mean()) if bets > 0 else 0.0
    return {
        "scenario": name,
        "bets": bets,
        "cost": round(cost, 2),
        "profit": round(profit, 2),
        "roi": round(roi, 6),
        "hit_rate": round(hit_rate, 6),
        "avg_ev_ratio": round(avg_ev, 6),
        "avg_confidence": round(avg_conf, 6),
        "avg_composite_score": round(avg_score, 4),
    }


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"Missing input: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        print(f"Input empty: {INPUT_CSV}")
        return

    df = df.copy()
    df["ev_ratio"] = _get_numeric_column(df, "ev_ratio", default=0.0)
    df["confidence"] = _get_numeric_column(df, "confidence", default=0.0)
    df["ruin_probability"] = _get_numeric_column(df, "ruin_probability", default=0.0)
    df["cost_if_buy"] = _pick_cost_column(df)
    df["profit_if_buy"] = _pick_profit_column(df)

    df["confidence_calibrated"] = df["confidence"].apply(calibrate_confidence)
    df["estimated_hit_rate"] = df["confidence"].apply(estimate_hit_probability)
    df["required_hit_rate"] = df.apply(
        lambda r: required_hit_probability(
            r.get("ev_ratio", 0.0), r.get("estimated_hit_rate", 0.0)
        ),
        axis=1,
    )
    df["probability_edge"] = df.apply(
        lambda r: probability_edge(r.get("ev_ratio", 0.0), r.get("confidence", 0.0)),
        axis=1,
    )
    df["risk_adjusted_ev"] = df.apply(
        lambda r: compute_risk_adjusted_ev(
            r.get("ev_ratio", 0.0), r.get("confidence", 0.0)
        ),
        axis=1,
    )
    df["composite_score"] = df.apply(
        lambda r: compute_composite_score(
            r.get("ev_ratio", 0.0),
            r.get("confidence", 0.0),
            r.get("ruin_probability", 0.0),
            r.get("risk_adjusted_ev", 0.0),
        ),
        axis=1,
    )
    df["grade"] = df.apply(
        lambda r: apply_grade_veto(
            compute_grade(r.get("composite_score", 0.0)),
            r.get("risk_adjusted_ev", 0.0),
            r.get("ruin_probability", 0.0),
            r.get("confidence", 0.0),
            r.get("ev_ratio", 0.0),
        ),
        axis=1,
    )

    meta_threshold = _load_meta_threshold(default=0.65)
    # confidence が tiny scale の場合に備えて、miss_proba は calibrated confidence から作る
    df["miss_proba_proxy"] = 1.0 - df["confidence_calibrated"].clip(0.0, 1.0)

    df["buy_baseline"] = df["ev_ratio"] >= float(EV_THRESHOLD_DEFAULT)
    df["buy_improved"] = (
        (df["ev_ratio"] >= float(EV_THRESHOLD_DEFAULT))
        & (df["miss_proba_proxy"] < float(meta_threshold))
        & (df["grade"] != "E")
    )

    baseline_summary = _summarize("baseline_ev_only", df, "buy_baseline")
    improved_summary = _summarize("improved_meta_gate", df, "buy_improved")

    delta = {
        "scenario": "delta_improved_minus_baseline",
        "bets": int(improved_summary["bets"] - baseline_summary["bets"]),
        "cost": round(
            float(improved_summary["cost"]) - float(baseline_summary["cost"]),
            2,
        ),
        "profit": round(
            float(improved_summary["profit"]) - float(baseline_summary["profit"]),
            2,
        ),
        "roi": round(
            float(improved_summary["roi"]) - float(baseline_summary["roi"]),
            6,
        ),
        "hit_rate": round(
            float(improved_summary["hit_rate"]) - float(baseline_summary["hit_rate"]),
            6,
        ),
        "avg_ev_ratio": round(
            float(improved_summary["avg_ev_ratio"])
            - float(baseline_summary["avg_ev_ratio"]),
            6,
        ),
        "avg_confidence": round(
            float(improved_summary["avg_confidence"])
            - float(baseline_summary["avg_confidence"]),
            6,
        ),
        "avg_composite_score": round(
            float(improved_summary["avg_composite_score"])
            - float(baseline_summary["avg_composite_score"]),
            4,
        ),
    }

    summary_df = pd.DataFrame([baseline_summary, improved_summary, delta])
    summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    detail_cols = [
        c
        for c in [
            "draw_date",
            "round_idx",
            "recommended",
            "ev_ratio",
            "confidence",
            "confidence_calibrated",
            "estimated_hit_rate",
            "required_hit_rate",
            "probability_edge",
            "miss_proba_proxy",
            "risk_adjusted_ev",
            "composite_score",
            "grade",
            "cost_if_buy",
            "profit_if_buy",
            "buy_baseline",
            "buy_improved",
        ]
        if c in df.columns
    ]
    df[detail_cols].to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

    print("=== E2E Revalidation Complete ===")
    print(f"meta_threshold_used: {meta_threshold:.4f}")
    print(f"saved_summary: {OUT_SUMMARY}")
    print(f"saved_detail: {OUT_DETAIL}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
