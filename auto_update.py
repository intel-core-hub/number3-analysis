import logging
import sys
import os
import requests
import pandas as pd
from numbers3_logic import (
    update_numbers3_clean,
    validate_numbers3,
    Numbers3Predictor,
    Numbers3MLPredictor,
)
from pathlib import Path
from datetime import datetime

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

    BASE_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    try:
        # 1. データ更新
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

        # 3. ハイブリッド予測
        logging.info("Generating hybrid predictions...")
        predictor = Numbers3Predictor(df)
        hybrid_pred_df = predictor.predict(
            top_n=10, model="hybrid", verbose=False
        )

        if hybrid_pred_df is None or hybrid_pred_df.empty:
            raise RuntimeError("Hybrid prediction DataFrame is empty")

        latest_pred_path = RESULTS_DIR / "latest_prediction.csv"
        hybrid_pred_df.to_csv(latest_pred_path, index=False)
        logging.info(f"Hybrid prediction saved to {latest_pred_path}")

        top_prediction = str(hybrid_pred_df.iloc[0]["予測番号"])
        recommended_numbers = [
            str(x) for x in hybrid_pred_df["予測番号"].tolist()
        ]

        # 4. LightGBM 予測（A-1.5の核心）
        ml_result = None
        ml_pred_df = None

        try:
            logging.info("Generating LightGBM prediction...")
            ml_predictor = Numbers3MLPredictor(df)
            ml_predictor.train()
            ml_result = ml_predictor.predict_next()

            predicted_digits = [int(d) for d in ml_result]

            ml_pred_df = pd.DataFrame([{
                "date": datetime.now().strftime("%Y-%m-%d"),
                "hundreds": predicted_digits[0],
                "tens": predicted_digits[1],
                "ones": predicted_digits[2],
                "model": "lightgbm"
            }])

            ml_output_path = RESULTS_DIR / "lightgbm_predictions.csv"
            ml_pred_df.to_csv(
                ml_output_path,
                mode="a",
                header=not ml_output_path.exists(),
                index=False,
                encoding="utf-8-sig",
            )

            logging.info(
                f"LightGBM prediction appended to {ml_output_path}"
            )

        except Exception as ml_error:
            logging.warning("ML prediction failed: %s", ml_error)

        # 5. Slack メッセージ
        latest_round = df.iloc[-1]
        ml_text = ml_result if ml_result else "N/A"

        msg = (
            f"✅ *ナンバーズ3 データ更新完了*\n"
            f"最新回号: 第{int(latest_round['回号'])}回\n"
            f"当選番号: {latest_round['当選番号']}\n"
            f"----------------------------\n"
            f"🔮 *次回（第{int(latest_round['回号']) + 1}回）のAI予測*\n"
            f"【第1候補】: *{top_prediction}*\n"
            f"【推奨数字】: {', '.join(recommended_numbers)}\n"
            f"🤖 *機械学習(LGBM)予測*: *{ml_text}*\n"
            f"----------------------------\n"
            f"詳細はこちら: https://github.com/{os.getenv('GITHUB_REPOSITORY')}"
        )

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
