from pathlib import Path
import sys

# Ensure project root is on sys.path so we can import `app` when running from tools/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import app helpers
from app import (
    DATA_PATH,
    EV_THRESHOLD_DEFAULT,
    load_predictions,
    prepare_payload_for_destination,
    post_webhook,
)

# Reuse the webhook constant from test file to avoid hardcoding twice
from tools.test_slack_post import WEBHOOK_URL


def main():
    df = load_predictions(str(DATA_PATH))
    if df.empty:
        print("No prediction data available.")
        sys.exit(1)

    required_columns = {"item_id", "ev_ratio", "ruin_probability", "confidence", "predicted_value"}
    missing = required_columns - set(df.columns)
    if missing:
        print("Missing columns:", missing)
        sys.exit(2)

    df = df.copy()
    for col in ["ev_ratio", "ruin_probability", "confidence", "predicted_value"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    eligible_df = df[df["ev_ratio"] >= float(EV_THRESHOLD_DEFAULT)].sort_values(["ev_ratio", "confidence"], ascending=[False, False])
    if eligible_df.empty:
        print("No eligible items for default EV threshold.")
        sys.exit(0)

    selected_row = eligible_df.iloc[0]

    payload = {
        "item_id": selected_row["item_id"],
        "ev_ratio": float(selected_row["ev_ratio"]),
        "confidence": float(selected_row["confidence"]),
        "ruin_probability": float(selected_row["ruin_probability"]),
        "decision": "buy_candidate",
    }

    headers = {"Content-Type": "application/json"}

    prepared = prepare_payload_for_destination(WEBHOOK_URL, payload)
    ok, message = post_webhook(prepared, WEBHOOK_URL, headers=headers)
    print("OK:", ok)
    print("MESSAGE:", message)


if __name__ == '__main__':
    main()
