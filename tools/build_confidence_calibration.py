from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "artifacts" / "historical_backtest_1y.csv"
OUTPUT_JSON = ROOT / "artifacts" / "confidence_calibration.json"


def _monotonic_non_decreasing(values: list[float]) -> list[float]:
    if not values:
        return values
    out = [values[0]]
    for value in values[1:]:
        out.append(max(out[-1], value))
    return out


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"Missing: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    if df.empty or "confidence" not in df.columns or "profit" not in df.columns:
        print("Required columns missing or input empty")
        return

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["confidence"])
    if df.empty:
        print("No valid confidence rows")
        return

    df = df.sort_values("confidence").reset_index(drop=True)
    df["hit"] = (df["profit"] > 0).astype(int)

    # 10 quantile bins (drop duplicates if low cardinality)
    df["bin"] = pd.qcut(df["confidence"], q=10, duplicates="drop")

    grouped = (
        df.groupby("bin", observed=False)
        .agg(
            n=("hit", "size"),
            hits=("hit", "sum"),
            conf_min=("confidence", "min"),
            conf_max=("confidence", "max"),
            conf_mean=("confidence", "mean"),
        )
        .reset_index(drop=True)
    )

    # Laplace smoothing to avoid hard 0/1
    grouped["hit_rate_raw"] = grouped["hits"] / grouped["n"]
    grouped["hit_rate_smooth"] = (grouped["hits"] + 1.0) / (grouped["n"] + 2.0)

    # Force monotonic non-decreasing mapping by confidence bin
    smooth_values = grouped["hit_rate_smooth"].tolist()
    mono_values = _monotonic_non_decreasing(smooth_values)
    grouped["hit_rate_calibrated"] = mono_values

    base_hit_rate = float(df["hit"].mean())
    max_hit_rate = float(grouped["hit_rate_calibrated"].max())

    bins = []
    for _, row in grouped.iterrows():
        bins.append(
            {
                "conf_min": float(row["conf_min"]),
                "conf_max": float(row["conf_max"]),
                "conf_mean": float(row["conf_mean"]),
                "n": int(row["n"]),
                "hits": int(row["hits"]),
                "hit_rate_raw": float(row["hit_rate_raw"]),
                "hit_rate_calibrated": float(row["hit_rate_calibrated"]),
            }
        )

    payload = {
        "source": str(INPUT_CSV),
        "rows": int(len(df)),
        "base_hit_rate": float(base_hit_rate),
        "max_hit_rate": float(max_hit_rate),
        "bins": bins,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote calibration: {OUTPUT_JSON}")
    print(
        f"rows={payload['rows']} base_hit_rate={base_hit_rate:.6f} max_hit_rate={max_hit_rate:.6f} bins={len(bins)}"
    )


if __name__ == "__main__":
    main()
