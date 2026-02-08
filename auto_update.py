import logging
import sys
import os
import requests
from numbers3_logic import update_numbers3_clean, validate_numbers3, Numbers3Predictor # クラス名を修正

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def send_slack_message(webhook_url: str, text: str):
    if not webhook_url:
        logging.warning("SLACK_WEBHOOK_URL is not set. Skipping notification.")
        return
    
    payload = {"text": text}
    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        logging.info("Slack notification sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send Slack notification: {e}")

def main() -> int:
    logging.info("Numbers3 data update starting...")
    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    
    try:
        # 1. データの自動更新
        df = update_numbers3_clean(
            clean_path="numbers3_clean.csv",
            backup=True,
            sleep_seconds=1.0,
            force_full=False,
        )
        
        # 2. バリデーション
        errors = validate_numbers3(df, strict=False, allow_missing_rounds=True)
        if errors:
            logging.warning("Update finished with warnings: %s", errors)

        # 3. 予測の実行 (ロジックに合わせて修正)
        logging.info("Generating predictions...")
        predictor = Numbers3Predictor(df)
        # ハイブリッドモデルで上位10個を予測
        pred_df = predictor.predict(top_n=10, model="hybrid", verbose=False)
        
        top_prediction = pred_df.iloc[0]["予測番号"]
        recommended_numbers = pred_df["予測番号"].tolist()
        
        # 4. Slackメッセージの構築
        latest_round = df.iloc[-1]
        msg = (
            f"✅ *ナンバーズ3 データ更新完了*\n"
            f"最新回号: 第{int(latest_round['回号'])}回\n"
            f"当選番号: {latest_round['当選番号']}\n"
            f"----------------------------\n"
            f"🔮 *次回（第{int(latest_round['回号'])+1}回）のAI予測*\n"
            f"【第1候補】: *{top_prediction}*\n"
            f"【推奨数字】: {', '.join(recommended_numbers)}\n"
            f"----------------------------\n"
            f"詳細はこちら: https://github.com/{os.getenv('GITHUB_REPOSITORY')}"
        )
        
        # 5. Slack通知の送信
        send_slack_message(slack_url, msg)
        
        logging.info("Update and notification finished successfully.")
        return 0
        
    except Exception:
        logging.exception("Update failed")
        error_msg = "❌ ナンバーズ3の自動更新中にエラーが発生しました。"
        send_slack_message(slack_url, error_msg)
        return 1

if __name__ == "__main__":
    sys.exit(main())