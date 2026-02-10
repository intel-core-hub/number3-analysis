#!/usr/bin/env python3
"""
Numbers3 機械学習バックテスト実行スクリプト（精度追跡対応版）

使用例:
    # 基本的な実行（精度追跡なし）
    python ml_backtest_runner.py --rounds 50
    
    # 精度追跡を有効にして実行
    python ml_backtest_runner.py --track-performance --rounds 100
    
    # ハイパーパラメータチューニングを10回ごとに実行
    python ml_backtest_runner.py --track-performance --rounds 100 --tune-every 10
    
    # ウィンドウサイズやブースティング回数を変更
    python ml_backtest_runner.py --track-performance --window 400 --boost-rounds 150 --rounds 50
"""

import argparse
import sys
import os
from datetime import datetime

import pandas as pd

from numbers3_logic import Numbers3MLBacktester

def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Numbers3 機械学習バックテスト（精度追跡対応）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  基本的な実行:
    python ml_backtest_runner.py --rounds 50
  
  精度追跡を有効にして実行:
    python ml_backtest_runner.py --track-performance --rounds 100
  
  ハイパーパラメータチューニング付き:
    python ml_backtest_runner.py --track-performance --rounds 100 --tune-every 10
        """
    )
    
    # 必須ではないオプション
    parser.add_argument(
        "--csv",
        default="numbers3_clean.csv",
        help="Numbers3データCSVのパス（デフォルト: numbers3_clean.csv）"
    )
    
    parser.add_argument(
        "--window",
        type=int,
        default=300,
        help="学習ウィンドウサイズ（デフォルト: 300）"
    )
    
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="バックテストの実行回数（デフォルト: 50）"
    )
    
    parser.add_argument(
        "--valid-size",
        type=int,
        default=180,
        help="ハイパーパラメータチューニング用の検証データサイズ（デフォルト: 180）"
    )
    
    parser.add_argument(
        "--tune-every",
        type=int,
        default=0,
        help="N回ごとにハイパーパラメータチューニングを実行（0で無効、デフォルト: 0）"
    )
    
    parser.add_argument(
        "--boost-rounds",
        type=int,
        default=120,
        help="LightGBMのブースティング回数（デフォルト: 120）"
    )
    
    parser.add_argument(
        "--output",
        default="results/ml_backtest_results.csv",
        help="バックテスト結果の出力先CSV（デフォルト: results/ml_backtest_results.csv）"
    )
    
    # 精度追跡関連
    parser.add_argument(
        "--track-performance",
        action="store_true",
        help="精度追跡を有効にする（推奨）"
    )
    
    parser.add_argument(
        "--performance-csv",
        default="results/ml_performance_tracking.csv",
        help="精度追跡CSVの出力先（デフォルト: results/ml_performance_tracking.csv）"
    )
    
    # その他
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細なログを出力"
    )
    
    return parser.parse_args()

def validate_inputs(args, df):
    """入力データとパラメータの妥当性チェック"""
    errors = []
    
    # データの行数チェック
    if len(df) < args.window + 10:
        errors.append(
            f"データが不足しています。最低 {args.window + 10} 行必要ですが、"
            f"{len(df)} 行しかありません。"
        )
    
    # ウィンドウサイズとバックテスト回数の整合性
    if args.rounds > len(df) - 2:
        errors.append(
            f"バックテスト回数（{args.rounds}）がデータ量に対して多すぎます。"
            f"最大 {len(df) - 2} 回まで可能です。"
        )
    
    # tune_everyの妥当性
    if args.tune_every > args.rounds:
        errors.append(
            f"--tune-every（{args.tune_every}）は --rounds（{args.rounds}）以下である必要があります。"
        )
    
    return errors

def print_summary(summary_df, args):
    """サマリー情報を整形して出力"""
    print("\n" + "="*70)


def save_performance_metric(summary_df, args):
    """精度追跡メトリクスをCSVに保存"""
    summary = summary_df.iloc[0]

    tracking_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tested_rounds": summary["tested"],
        "straight_hit_rate": summary["set_straight_hits"] / summary["tested"],
        "box_hit_rate": summary["set_box_hits"] / summary["tested"],
        "roi": summary["roi_pct"],
        "window_size": args.window,
        "boost_rounds": args.boost_rounds,
        "total_profit": summary["total_profit"],
    }

    new_row = pd.DataFrame([tracking_data])

    os.makedirs(os.path.dirname(args.performance_csv), exist_ok=True)
    header = not os.path.exists(args.performance_csv)
    new_row.to_csv(
        args.performance_csv,
        mode="a",
        index=False,
        header=header,
        encoding="utf-8-sig",
    )
    print(f"  ✓ 精度追跡データ: {args.performance_csv}")
    print("📊 バックテスト結果サマリー")
    print("="*70)
    
    summary = summary_df.iloc[0]
    
    print(f"\n【実行パラメータ】")
    print(f"  学習ウィンドウ: {args.window}回")
    print(f"  ブースティング回数: {args.boost_rounds}回")
    print(f"  テスト回数: {summary['tested']}回")
    
    print(f"\n【的中結果】")
    print(f"  セット・ストレート的中: {summary['set_straight_hits']}回")
    print(f"  セット・ボックス的中: {summary['set_box_hits']}回")
    print(f"  的中率（ストレート）: {summary['set_straight_hits'] / summary['tested'] * 100:.2f}%")
    print(f"  的中率（ボックス）: {summary['set_box_hits'] / summary['tested'] * 100:.2f}%")
    
    print(f"\n【収益性】")
    print(f"  総投資額: {summary['total_cost']:,}円")
    print(f"  総払戻額: {summary['total_return']:,}円")
    print(f"  損益: {summary['total_profit']:+,}円")
    print(f"  ROI: {summary['roi_pct']:+.2f}%")
    
    print("\n" + "="*70)

def main():
    # コマンドライン引数のパース
    args = parse_arguments()
    
    # 開始時刻
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("🚀 Numbers3 機械学習バックテスト")
    print("="*70)
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # データ読み込み
    try:
        print(f"\n📂 データ読み込み中: {args.csv}")
        df = pd.read_csv(args.csv)
        print(f"  ✓ {len(df)}行のデータを読み込みました")
    except FileNotFoundError:
        print(f"\n❌ エラー: {args.csv} が見つかりません。")
        print("   データを更新してから再実行してください。")
        return 1
    except Exception as e:
        print(f"\n❌ データ読み込みエラー: {e}")
        return 1
    
    # 入力バリデーション
    validation_errors = validate_inputs(args, df)
    if validation_errors:
        print("\n❌ 入力エラー:")
        for error in validation_errors:
            print(f"  - {error}")
        return 1
    
    # tune_everyの処理
    tune_every = args.tune_every if args.tune_every > 0 else None
    
    # バックテスターのインスタンス化
    print(f"\n⚙️  バックテスター設定中...")
    print(f"  学習ウィンドウ: {args.window}回")
    print(f"  テスト回数: {args.rounds}回")
    print(f"  ブースティング回数: {args.boost_rounds}回")
    if tune_every:
        print(f"  ハイパーパラメータチューニング: {tune_every}回ごと")
    else:
        print(f"  ハイパーパラメータチューニング: 無効")
    
    if args.track_performance:
        print(f"  📊 精度追跡: 有効")
        print(f"     保存先: {args.performance_csv}")
    else:
        print(f"  📊 精度追跡: 無効")
    
    try:
        backtester = Numbers3MLBacktester(
            df,
            window=args.window,
            test_rounds=args.rounds,
            valid_size=args.valid_size,
            tune_every=tune_every,
            num_boost_round=args.boost_rounds,
            track_performance=args.track_performance,
            performance_csv=args.performance_csv,
        )
    except Exception as e:
        print(f"\n❌ バックテスター初期化エラー: {e}")
        return 1
    
    # バックテスト実行
    print(f"\n🔄 バックテスト実行中...")
    print(f"   （これには数分かかる場合があります）\n")
    
    try:
        results_df, summary_df = backtester.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  ユーザーによって中断されました。")
        return 1
    except Exception as e:
        print(f"\n❌ バックテスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 結果の保存
    print(f"\n💾 結果を保存中...")
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"  ✓ バックテスト結果: {args.output}")
    except Exception as e:
        print(f"  ❌ 保存エラー: {e}")
        return 1
    
    # サマリー表示
    print_summary(summary_df, args)

    if args.track_performance:
        try:
            save_performance_metric(summary_df, args)
        except Exception as e:
            print(f"  ⚠️  精度追跡データ保存エラー: {e}")
    
    # 終了時刻と所要時間
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\n終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"所要時間: {elapsed.total_seconds():.1f}秒")
    
    # 次のステップの案内
    if args.track_performance:
        print(f"\n💡 次のステップ:")
        print(f"   精度分析を行うには、以下のコマンドを実行してください:")
        print(f"   jupyter notebook notebooks/analysis_ml_performance.ipynb")
    
    print("\n✅ バックテスト完了\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())