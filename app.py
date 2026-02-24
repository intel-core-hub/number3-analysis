from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


DATA_PATH = Path("data/processed/latest_predictions.parquet")
EV_THRESHOLD_DEFAULT = 1.05
RUIN_WARNING_THRESHOLD = 0.20
RUIN_ERROR_THRESHOLD = 0.35
CONFIDENCE_WARNING_THRESHOLD = 0.60
CONFIDENCE_ERROR_THRESHOLD = 0.45


st.set_page_config(page_title="予測・購買判断システム", page_icon="🎯", layout="wide")


@st.cache_data(ttl=1800, show_spinner=False)
def load_predictions(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()

    if file_path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def post_webhook(payload: dict[str, Any], webhook_url: str) -> tuple[bool, str]:
    if not webhook_url:
        return False, "Webhook URL が未設定です"
    try:
        response = requests.post(webhook_url, json=payload, timeout=8)
        if 200 <= response.status_code < 300:
            return True, f"Webhook送信成功: {response.status_code}"
        return False, f"Webhook送信失敗: {response.status_code}"
    except Exception as exc:
        return False, f"Webhookエラー: {exc}"


def ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    target_columns = ["ev_ratio", "ruin_probability", "confidence", "predicted_value"]
    for column in target_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def render_risk_alerts(row: pd.Series) -> None:
    ruin_probability = float(row.get("ruin_probability", 0.0) or 0.0)
    confidence = float(row.get("confidence", 0.0) or 0.0)

    if ruin_probability >= RUIN_ERROR_THRESHOLD:
        st.error(f"破綻確率が高いです（{ruin_probability:.1%}）。購入を再検討してください。")
    elif ruin_probability >= RUIN_WARNING_THRESHOLD:
        st.warning(f"破綻確率に注意が必要です（{ruin_probability:.1%}）。")

    if confidence <= CONFIDENCE_ERROR_THRESHOLD:
        st.error(f"モデル信頼度が低いです（{confidence:.1%}）。")
    elif confidence <= CONFIDENCE_WARNING_THRESHOLD:
        st.warning(f"モデル信頼度がやや低めです（{confidence:.1%}）。")


def main() -> None:
    st.title("🎯 予測・購買判断システム（本番）")
    st.caption("本アプリは事前計算済み結果のみを表示し、推論処理は実行しません。")

    if "selected_item_id" not in st.session_state:
        st.session_state.selected_item_id = None
    if "last_action_message" not in st.session_state:
        st.session_state.last_action_message = ""

    with st.sidebar:
        st.subheader("運用設定")
        ev_threshold = st.number_input(
            "EV閾値（ハードリミット）",
            min_value=1.00,
            max_value=2.00,
            value=EV_THRESHOLD_DEFAULT,
            step=0.01,
            help="この閾値未満の案件は一覧に表示しません。",
        )
        webhook_url = st.text_input("Webhook URL（任意）", value="", type="password")
        if st.button("データ再読込", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    df = load_predictions(str(DATA_PATH))
    if df.empty:
        st.error("予測データがありません。バッチ処理の出力を確認してください。")
        st.stop()

    required_columns = {"item_id", "ev_ratio", "ruin_probability", "confidence", "predicted_value"}
    missing = required_columns - set(df.columns)
    if missing:
        st.error(f"必要列が不足しています: {', '.join(sorted(missing))}")
        st.stop()

    df = ensure_numeric_columns(df.copy())
    df = df.dropna(subset=["ev_ratio", "ruin_probability", "confidence"])

    eligible_df = df[df["ev_ratio"] >= float(ev_threshold)].copy()
    eligible_df = eligible_df.sort_values(["ev_ratio", "confidence"], ascending=[False, False])

    if eligible_df.empty:
        st.info("本日は推奨案件がありません。")
        st.caption("購買基準を満たす案件がないため、以降の表示をスキップします。")
        st.stop()

    st.success(f"推奨案件: {len(eligible_df)}件")

    display_columns = [
        "item_id",
        "predicted_value",
        "ev_ratio",
        "confidence",
        "ruin_probability",
    ]
    if "updated_at" in eligible_df.columns:
        display_columns.append("updated_at")

    st.dataframe(
        eligible_df[display_columns],
        width="stretch",
        hide_index=True,
    )

    selected_item = st.selectbox(
        "購買対象の案件を選択",
        options=eligible_df["item_id"].tolist(),
        index=0,
    )

    selected_row = eligible_df[eligible_df["item_id"] == selected_item].iloc[0]
    st.session_state.selected_item_id = selected_item

    c1, c2, c3 = st.columns(3)
    c1.metric("EV", f"{float(selected_row['ev_ratio']):.2f}")
    c2.metric("信頼度", f"{float(selected_row['confidence']):.1%}")
    c3.metric("破綻確率", f"{float(selected_row['ruin_probability']):.1%}")

    render_risk_alerts(selected_row)

    if st.button("この案件を購入判断として送信", type="primary", use_container_width=True):
        payload = {
            "item_id": selected_row["item_id"],
            "ev_ratio": float(selected_row["ev_ratio"]),
            "confidence": float(selected_row["confidence"]),
            "ruin_probability": float(selected_row["ruin_probability"]),
            "decision": "buy_candidate",
        }
        ok, message = post_webhook(payload, webhook_url)
        st.session_state.last_action_message = message
        if ok:
            st.success(message)
        else:
            st.warning(message)

    if st.session_state.last_action_message:
        st.caption(st.session_state.last_action_message)


if __name__ == "__main__":
    main()