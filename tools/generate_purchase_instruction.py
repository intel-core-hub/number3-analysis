from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (
    DATA_PATH,
    EV_THRESHOLD_DEFAULT,
    load_predictions,
    ensure_numeric_columns,
    compute_risk_adjusted_ev,
    compute_kelly_fraction,
)


def run(capital: float = 10000.0, risk_pct: float = 0.01):
    df = load_predictions(str(DATA_PATH))
    if df.empty:
        print("No data")
        return

    df = ensure_numeric_columns(df.copy())
    df = df.dropna(subset=["ev_ratio", "ruin_probability", "confidence"])
    eligible = df[df["ev_ratio"] >= EV_THRESHOLD_DEFAULT].copy()
    if eligible.empty:
        print("No eligible items")
        return

    eligible["risk_adjusted_ev"] = eligible.apply(lambda r: compute_risk_adjusted_ev(r.get("ev_ratio", 0.0), r.get("confidence", 0.0)), axis=1)
    top = eligible.sort_values(["risk_adjusted_ev", "ev_ratio"], ascending=[False, False]).iloc[0]

    invest_amount = capital * risk_pct
    predicted_price = float(top.get("predicted_value") or 0.0)
    est_qty = None
    if predicted_price > 0:
        est_qty = invest_amount / predicted_price

    print("Top item:", top.get("item_id"))
    print(f"EV={top.get('ev_ratio')}, confidence={top.get('confidence')}, ruin={top.get('ruin_probability')}")
    print(f"Risk-adjusted EV={top.get('risk_adjusted_ev')}")
    print(f"Capital={capital}, risk%={risk_pct}, invest_amount={invest_amount}")
    if est_qty is not None:
        print(f"Estimated quantity (using predicted_value): {est_qty:.4f} @ {predicted_price:.2f}")
    else:
        print("No predicted price available — cannot estimate quantity.")
    # Kelly
    kelly = compute_kelly_fraction(top.get("ev_ratio"), top.get("confidence"), loss_fraction=1.0)
    print(f"Kelly fraction (approx): {kelly:.3%}")
    print(f"Kelly invest amount: {capital * kelly:.2f}")


if __name__ == '__main__':
    run(10000.0, 0.01)
