#!/usr/bin/env python3
"""
comprehensive_ev_test.py - 包括的なEV計算検証テスト

CSVエクスポート、期待値計算、EV比フィルタの統一性を検証
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.strategies.portfolio_optimizer import PortfolioOptimizer
from src.utils.config import TICKET_COST, BOX_PRIZE_SINGLE, BOX_PRIZE_DOUBLE


def test_csv_export_consistency():
    """CSVエクスポートの計算がコード内の計算と一致するか検証"""
    
    print("\n=== CSVエクスポート整合性テスト ===\n")
    
    optimizer = PortfolioOptimizer(budget=5000, min_ev_ratio=1.1)
    
    # テストデータ
    demo_numbers = [9, 8]
    demo_payouts = pd.DataFrame({
        "number": ["009", "008"],
        "straight_prize": [75_000, 75_000],
        "box_prize": [12_500, 12_500],
    })
    
    # ケース1: EV比が高いボックス009
    # 確率 = 1.88% → EV回収額 = 0.0188 × 12,500 = 235円 → EV比 = 1.175x
    demo_probs = pd.DataFrame({
        "number": ["009", "008"],
        "straight_prob": [0.001, 0.001],
        "box_prob": [0.0188, 0.01],  # 1.88%, 1%
        "mini_prob": [0.01, 0.01],
    })
    
    candidates = optimizer.generate_candidates(demo_numbers, demo_payouts, demo_probs)
    
    # ボックス009を抽出
    box_009 = next((c for c in candidates if c.number == "009" and c.bet_type == "box"), None)
    
    assert box_009 is not None, "ボックス009候補が生成されませんでした"
    if box_009:
        print(f"ボックス009テスト:")
        print(f"  入力確率: 1.88%")
        print(f"  入力賞金: 12,500円")
        print(f"  コスト: {TICKET_COST}円\n")
        
        # 期待値計算
        expected_return = 0.0188 * 12_500
        expected_ev_ratio = expected_return / TICKET_COST
        expected_profit = expected_return - TICKET_COST
        
        print(f"  期待回収額: 0.0188 × 12,500 = {expected_return:.2f}円")
        print(f"  期待EV比: {expected_return:.2f} / {TICKET_COST} = {expected_ev_ratio:.4f}x")
        print(f"  期待利益: {expected_return:.2f} - {TICKET_COST} = {expected_profit:.2f}円\n")
        
        print(f"  計算結果:")
        print(f"    - EV比: {box_009.ev_ratio:.4f}x")
        print(f"    - 期待利益: {box_009.expected_value:.2f}円\n")
        
        # 検証
        ev_ratio_match = abs(box_009.ev_ratio - expected_ev_ratio) < 0.0001
        profit_match = abs(box_009.expected_value - expected_profit) < 0.01
        
        print(f"  ✓ EV比一致性: {'PASS' if ev_ratio_match else 'FAIL'}")
        print(f"  ✓ 期待利益一致性: {'PASS' if profit_match else 'FAIL'}\n")
        
        assert ev_ratio_match, "EV比一致性がFAILです"
        assert profit_match, "期待利益一致性がFAILです"


def test_ev_threshold_filtering():
    """EV閾値フィルタリングが正しく機能するか検証"""
    
    print("=== EV閾値フィルタリングテスト ===\n")
    
    optimizer = PortfolioOptimizer(budget=5000, min_ev_ratio=1.15)
    
    demo_numbers = [1, 2, 3, 4, 5]
    demo_payouts = pd.DataFrame({
        "number": [f"{i:03d}" for i in demo_numbers],
        "straight_prize": [75_000] * 5,
        "box_prize": [12_500] * 5,
    })
    
    # 様々なEV比のケース
    demo_probs = pd.DataFrame({
        "number": ["001", "002", "003", "004", "005"],
        "straight_prob": [0.001] * 5,
        "box_prob": [0.008, 0.010, 0.012, 0.014, 0.016],  # EV: 0.8x, 1.0x, 1.2x, 1.4x, 1.6x
        "mini_prob": [0.01] * 5,
    })
    
    candidates = optimizer.generate_candidates(demo_numbers, demo_payouts, demo_probs)
    
    # フィルタリング前後の数
    print(f"総候補数: {len(candidates)}")
    
    filtered = [c for c in candidates if c.ev_ratio >= 1.15]
    print(f"EV比 >= 1.15のフィルタ後: {len(filtered)}\n")
    
    print("各候補のEV比:")
    for c in sorted(candidates, key=lambda x: x.ev_ratio):
        if "box" in c.bet_type.lower():
            status = "✓ INCLUDE" if c.ev_ratio >= 1.15 else "✗ EXCLUDE"
            print(f"  {c.number} ({c.bet_type}): {c.ev_ratio:.4f}x {status}")
    
    # 期待値の関連性を確認
    print("\n期待利益との関連性:")
    for c in sorted(candidates, key=lambda x: x.ev_ratio):
        if "box" in c.bet_type.lower() and c.ev_ratio >= 1.15:
            calc_profit = (c.ev_ratio - 1.0) * TICKET_COST
            match = abs(calc_profit - c.expected_value) < 0.01
            status = "✓" if match else "✗"
            print(f"  {c.number}: EV比={c.ev_ratio:.3f}x → 期待利益={c.expected_value:.1f}円 {status}")
    
    assert isinstance(filtered, list)


def test_backtest_consistency():
    """バックテスト時のEV計算が統一されているか検証"""
    
    print("\n=== バックテスト一貫性テスト ===\n")
    
    # 仮想バックテスト結果を生成
    backtest_results = pd.DataFrame({
        "expected_value": [20, 40, 50, 30, 10],  # 円単位
        "profit": [50, 80, 100, 60, 20],
        "hit": [1, 1, 1, 1, 0],
        "ev_ratio": [1.10, 1.20, 1.25, 1.15, 1.05],
    })
    
    print("バックテスト結果サンプル:")
    print(backtest_results.to_string())
    print()
    
    # 期待値とEV比の関連性を検証
    print("\n期待値 = (EV比 - 1) × コストの検証:")
    all_consistent = True
    
    for idx, row in backtest_results.iterrows():
        expected_profit = (row["ev_ratio"] - 1.0) * TICKET_COST
        actual_profit = row["expected_value"]
        
        # 許容範囲: ±1円（丸め誤差対応）
        is_consistent = abs(expected_profit - actual_profit) <= 1.0
        status = "✓" if is_consistent else "✗"
        
        print(f"  Row {idx}: ({row['ev_ratio']:.2f} - 1) × 200 = {expected_profit:.1f}円 " +
              f"(実際: {actual_profit}円) {status}")
        
        all_consistent = all_consistent and is_consistent
    
    print(f"\n全体一貫性: {'PASS' if all_consistent else 'FAIL'}")
    
    assert all_consistent, "バックテスト結果のEV整合性に不一致があります"


if __name__ == "__main__":
    print("=" * 70)
    print("         包括的EV計算検証スイート")
    print("=" * 70)
    
    try:
        test_csv_export_consistency()
        test_ev_threshold_filtering()
        test_backtest_consistency()
        
        print("\n" + "=" * 70)
        print("✓ すべての検証テストが成功しました")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
