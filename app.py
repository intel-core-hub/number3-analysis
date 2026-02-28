from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import uuid
import hmac
import hashlib
import json

import logging
import pandas as pd
import requests
import streamlit as st


DATA_PATH = Path("data/processed/latest_predictions.parquet")
EV_THRESHOLD_DEFAULT = 1.05
RUIN_WARNING_THRESHOLD = 0.20
RUIN_ERROR_THRESHOLD = 0.35
CONFIDENCE_WARNING_THRESHOLD = 0.60
CONFIDENCE_ERROR_THRESHOLD = 0.45
RUIN_FILTER_DEFAULT = 0.10
CAPITAL_DEFAULT = 10000.0
RISK_PCT_DEFAULT = 0.01


st.set_page_config(page_title="予測・購買判断システム", page_icon="🎯", layout="wide")

# webhook ログ設定
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
webhook_logger = logging.getLogger("webhook_logger")
if not webhook_logger.handlers:
    fh = logging.FileHandler(LOG_DIR / "webhook.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    webhook_logger.addHandler(fh)
    webhook_logger.setLevel(logging.INFO)


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


def post_webhook(
    payload: Any,
    webhook_url: str,
    headers: dict[str, str] | None = None,
    max_retries: int = 2,
    raw: bool = False,
) -> tuple[bool, str]:
    if not webhook_url:
        return False, "Webhook URL が未設定です"
    # デフォルトヘッダー
    headers = headers or {"Content-Type": "application/json"}
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if raw:
                # payload expected as str or bytes
                response = requests.post(webhook_url, data=payload, headers=headers, timeout=10)
            else:
                response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
        except requests.RequestException as exc:
            last_exc = exc
            webhook_logger.error("Attempt %d POST to %s failed: %s | payload=%s", attempt, webhook_url, exc, payload)
            if attempt == max_retries:
                return False, f"Webhookエラー: {exc}"
            continue

        status = response.status_code
        body = ""
        try:
            body = response.text or ""
        except Exception:
            body = "(response body unavailable)"

        # ログに詳細を残す
        webhook_logger.info("POST %s attempt=%d status=%d payload=%s response=%s", webhook_url, attempt, status, payload, body)

        if 200 <= status < 300:
            return True, f"Webhook送信成功: {status} | {body}"

        return False, f"Webhook送信失敗: {status} | {body}"

    # ここには来ないはずだが念のため
    if last_exc is not None:
        return False, f"Webhook例外: {last_exc}"
    return False, "Webhook送信失敗（不明な理由）"


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
 
def prepare_payload_for_destination(webhook_url: str, internal_payload: dict[str, Any]) -> Any:
    """
    Convert internal payload into a destination-specific format.

    Currently supports Slack incoming webhooks by returning a simple
    `{"text": "..."}` message. Returns the original payload by default.
    """
    if not webhook_url:
        return internal_payload

    lower = webhook_url.lower()
    # Slack incoming webhook
    if "hooks.slack.com" in lower:
        try:
            item = internal_payload.get("item_id")
            ev = float(internal_payload.get("ev_ratio", 0.0) or 0.0)
            conf = float(internal_payload.get("confidence", 0.0) or 0.0)
            ruin = float(internal_payload.get("ruin_probability", 0.0) or 0.0)
        except Exception:
            # best-effort stringification
            text = f"[numbers3] {internal_payload}"
            return {"text": text}

        text = (
            f"[numbers3] item:{item} EV:{ev:.2f} "
            f"信頼度:{conf:.1%} 破綻確率:{ruin:.1%}"
        )
        return {"text": text}

    # Default: return internal payload unchanged
    return internal_payload


def compute_risk_adjusted_ev(ev_ratio: float, confidence: float) -> float:
    """Compute simple risk-adjusted expected return per the user's request.

    Formula used (assumes incorrect prediction results in full loss):
        E = confidence * (ev_ratio - 1) + (1 - confidence) * (-1)

    Returns decimal (e.g. 0.01 == +1%).
    """
    try:
        ev = float(ev_ratio or 0.0)
        conf = float(confidence or 0.0)
    except Exception:
        return 0.0

    return conf * (ev - 1.0) + (1.0 - conf) * (-1.0)


def compute_kelly_fraction(ev_ratio: float, confidence: float, loss_fraction: float = 1.0) -> float:
    """Simple Kelly fraction estimate under binary win/loss model.

    - win payoff net b = ev_ratio - 1
    - if lose, lose `loss_fraction` of stake (default 1.0 == full stake lost)

    Uses formula adapted for asymmetric loss: approx f* = (p*(b+1) - 1) / (b + ( (1-p)/p )*loss_fraction )
    This is a heuristic. Clip to [0,1].
    """
    try:
        p = float(confidence or 0.0)
        b = float(ev_ratio or 0.0) - 1.0
    except Exception:
        return 0.0

    if b <= 0 or p <= 0:
        return 0.0

    # heuristic denominator to account for loss fraction
    denom = b + ((1.0 - p) / p) * loss_fraction
    if denom <= 0:
        return 0.0
    f = (p * (b + 1.0) - 1.0) / denom
    # clip
    return max(0.0, min(1.0, f))


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
        # Webhook URL: prefer environment variable `SLACK_WEBHOOK_URL` if present
        default_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
        webhook_url = st.text_input("Webhook URL (送信先)", value=default_webhook, type="password", help="例: Slack incoming webhook URL")
        webhook_auth = st.text_input(
            "Webhook Authorization ヘッダー（任意、例: Bearer <token>)",
            value="",
            type="password",
        )
        if st.button("テスト通知を送る", use_container_width=True):
            if not webhook_url:
                st.warning("Webhook URL が未設定です。sidebar の Webhook URL を入力してください。")
            else:
                test_payload = {"text": "[numbers3] UI テスト通知 from Streamlit"}
                prepared = prepare_payload_for_destination(webhook_url, test_payload)
                headers: dict[str, str] | None = {"Content-Type": "application/json"}
                ok, message = post_webhook(prepared, webhook_url, headers=headers)
                webhook_logger.info("UI test send webhook=%s ok=%s msg=%s", webhook_url, ok, message)
                if ok:
                    st.success(message)
                else:
                    st.error(message)
        # 自動フィルタ設定: 破綻確率上限
        st.divider()
        st.markdown("**自動フィルタ（破綻確率）**")
        ruin_filter_threshold = st.number_input(
            "破綻確率の上限（この値より大きい案件は自動除外）",
            min_value=0.0,
            max_value=1.0,
            value=RUIN_FILTER_DEFAULT,
            step=0.01,
            format="%.2f",
        )
        apply_ruin_filter = st.checkbox("破綻確率による自動除外を適用する", value=True)
        st.divider()
        st.markdown("**自動推薦と購入指示**")
        capital = st.number_input("運用資本 (通貨単位)", value=float(os.environ.get("NUMBERS3_CAPITAL", CAPITAL_DEFAULT)), step=100.0)
        risk_pct = st.number_input("1トレードあたりの許容リスク（資本比率）", min_value=0.0, max_value=1.0, value=RISK_PCT_DEFAULT, step=0.001, format="%.3f")
        auto_recommend = st.checkbox("自動で上位案件を推薦して購入指示を表示する", value=True)
        sizing_method = st.selectbox("ポジションサイズ方式", options=["固定% (許容リスク)", "Kelly (簡易)", "None"], index=0)
        loss_fraction = st.number_input("損失幅（ポジション比率、Kelly用、1.0=全額損失想定）", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
        emulate_github = st.checkbox("GitHub イベントをエミュレートして送信する（高度）", value=False)
        github_repo = st.text_input("エミュレート用リポジトリ名（owner/repo、任意）", value="")
        github_secret = st.text_input("GitHub webhook secret（任意、シグネチャ生成に使用）", value="", type="password")
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

    # Apply ruin-probability auto-filter if requested
    if apply_ruin_filter:
        before_count = len(eligible_df)
        eligible_df = eligible_df[eligible_df["ruin_probability"] <= float(ruin_filter_threshold)].copy()
        after_count = len(eligible_df)
        removed = before_count - after_count
        if removed > 0:
            st.info(f"破綻確率フィルタにより {removed} 件が除外されました（閾値: {ruin_filter_threshold:.2f}）。")

    if eligible_df.empty:
        st.info("本日は推奨案件がありません（フィルタ適用後）。")
        st.caption("購買基準を満たす案件がないため、以降の表示をスキップします。")
        st.stop()

    st.success(f"推奨案件: {len(eligible_df)}件")

    display_columns = [
        "item_id",
        "predicted_value",
        "risk_adjusted_ev",
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

    # Compute risk-adjusted EV column (decimal)
    eligible_df["risk_adjusted_ev"] = eligible_df.apply(
        lambda r: compute_risk_adjusted_ev(r.get("ev_ratio", 0.0), r.get("confidence", 0.0)), axis=1
    )

    # select row after computing new column
    selected_row = eligible_df[eligible_df["item_id"] == selected_item].iloc[0]
    st.session_state.selected_item_id = selected_item
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EV", f"{float(selected_row['ev_ratio']):.2f}")
    c2.metric("信頼度", f"{float(selected_row['confidence']):.1%}")
    c3.metric("破綻確率", f"{float(selected_row['ruin_probability']):.1%}")
    radv = float(selected_row.get("risk_adjusted_ev", 0.0) or 0.0)
    c4.metric("リスク調整期待値", f"{radv:.1%}")

    render_risk_alerts(selected_row)

    if st.button("この案件を購入判断として送信", type="primary", use_container_width=True):
        payload = {
            "item_id": selected_row["item_id"],
            "ev_ratio": float(selected_row["ev_ratio"]),
            "confidence": float(selected_row["confidence"]),
            "ruin_probability": float(selected_row["ruin_probability"]),
            "decision": "buy_candidate",
        }
        # ヘッダーを組み立て（Authorization があれば付与）
        headers: dict[str, str] | None = {"Content-Type": "application/json"}
        if webhook_auth:
            headers["Authorization"] = webhook_auth

        # GitHub イベントをエミュレートする場合は特別なヘッダー/ボディ整形
        if emulate_github:
            # 最小限の push イベントを作る
            push_event = {
                "ref": "refs/heads/main",
                "repository": {"full_name": github_repo or "unknown/unknown"},
                "pusher": {"name": "numbers3-app"},
                "head_commit": {"id": str(uuid.uuid4()), "message": "purchase-candidate"},
                # include our payload as custom field for receiver
                "numbers3_payload": payload,
            }
            body_str = json.dumps(push_event)
            # GitHub-like headers
            headers["X-GitHub-Event"] = "push"
            headers["X-GitHub-Delivery"] = str(uuid.uuid4())
            # If secret provided, compute X-Hub-Signature (sha1)
            if github_secret:
                sig = hmac.new(github_secret.encode("utf-8"), body_str.encode("utf-8"), hashlib.sha1).hexdigest()
                headers["X-Hub-Signature"] = f"sha1={sig}"
            # Send raw body (not JSON param) so signature matches
            ok, message = post_webhook(body_str, webhook_url, headers=headers, raw=True)
        else:
            prepared = prepare_payload_for_destination(webhook_url, payload)
            ok, message = post_webhook(prepared, webhook_url, headers=headers)
        st.session_state.last_action_message = message
        if ok:
            st.success(message)
        else:
            st.warning(message)

    # 自動推薦・購入指示の表示
    if auto_recommend:
        # pick top by risk_adjusted_ev
        top = eligible_df.sort_values(["risk_adjusted_ev", "ev_ratio"], ascending=[False, False]).iloc[0]
        st.markdown("**自動推薦（上位1件）**")
        st.write(top[display_columns + (["updated_at"] if "updated_at" in eligible_df.columns else [])])

        invest_amount = float(capital) * float(risk_pct)
        predicted_price = float(top.get("predicted_value") or 0.0)
        if predicted_price > 0:
            est_qty = invest_amount / predicted_price
            st.info(f"推奨投資額: {invest_amount:.2f}、概算数量: {est_qty:.4f}（予測価格 {predicted_price:.2f} を使用）")
        else:
            st.info(f"推奨投資額: {invest_amount:.2f}（予測価格がないため数量は算出できません）")

        st.markdown("**注文案（参考）**")
        st.write("- 注文タイプ: 成行（もしくは短期の指値を併用）")
        st.write("- 推奨投資額は資本の許容リスクに基づく目安です。実運用ではストップロスや取引手数料を考慮してください。")

        # Generate order template
        order = {
            "item_id": top.get("item_id"),
            "side": "buy",
            "quantity": round(float(est_qty) if est_qty is not None else 0.0, 6),
            "price": float(top.get("predicted_value") or 0.0),
            "order_type": "market",
            "notes": "Generated by numbers3 app (参考)。ストップと手数料を適宜設定してください。",
        }
        st.markdown("**生成された注文テンプレート（JSON）**")
        st.code(json.dumps(order, ensure_ascii=False, indent=2))

        # Confirm & send
        st.markdown("**注文のWebhook送信（任意・要確認）**")
        confirm_send = st.checkbox("上記の注文をWebhookで送信することを確認します（自己責任）")
        if confirm_send and st.button("Webhookで注文を送信", use_container_width=True):
            if not webhook_url:
                st.error("Webhook URL が未設定です。サイドバーにWebhook URLを入力してください。")
            else:
                # send order as JSON (do not transform to Slack text)
                headers_order = {"Content-Type": "application/json"}
                ok2, msg2 = post_webhook(order, webhook_url, headers=headers_order)
                if ok2:
                    st.success(f"注文テンプレートを送信しました: {msg2}")
                else:
                    st.error(f"送信失敗: {msg2}")

    if st.session_state.last_action_message:
        st.caption(st.session_state.last_action_message)


if __name__ == "__main__":
    main()