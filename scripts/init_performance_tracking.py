import pandas as pd
import os

# CSVのカラム定義
COLUMNS = [
    "timestamp",
    "backtest_id",
    "target_round",
    "actual_number",
    "predicted_number",
    "model_type",
    "window_size",
    "num_boost_round",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "straight_hit",
    "box_hit",
    "digit1_correct",
    "digit2_correct",
    "digit3_correct",
    "logloss_digit1",
    "logloss_digit2",
    "logloss_digit3",
    "logloss_avg",
    "confidence_digit1",
    "confidence_digit2",
    "confidence_digit3",
    "confidence_avg",
    "prize_amount",
    "profit",
    "cumulative_profit",
    "roi_percent",
]

def initialize_performance_tracking_csv(filepath="results/ml_performance_tracking.csv"):
    """
    精度追跡CSVを初期化（既存の場合は何もしない）
    """
    # ディレクトリがなければ作成
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # ファイルが既に存在する場合はスキップ
    if os.path.exists(filepath):
        print(f"✓ {filepath} already exists. Skipping initialization.")
        return
    
    # 空のDataFrameを作成してCSV保存
    df = pd.DataFrame(columns=COLUMNS)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"✓ Initialized {filepath}")

def main():
    print("\n" + "="*60)
    print("📊 精度追跡CSV 初期化")
    print("="*60 + "\n")
    
    initialize_performance_tracking_csv()
    
    print("\n✅ 初期化完了")
    print("\n次のステップ:")
    print("  1. データを更新: Streamlitアプリまたはスクレイピング")
    print("  2. バックテスト実行: python ml_backtest_runner.py --track-performance --rounds 50")
    print("  3. 結果分析: jupyter notebook notebooks/analysis_ml_performance.ipynb\n")

if __name__ == "__main__":
    main()