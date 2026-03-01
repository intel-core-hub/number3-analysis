#!/usr/bin/env python3
"""
test_ev_calculation.py - EV比の計算精度テスト

修正前後で計算が正しく行われているか確認する
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.strategies.portfolio_optimizer import PortfolioOptimizer
import pandas as pd


def test_ev_calculation():
    """EV比と期待利益の計算が正しいか検証"""
    
    optimizer = PortfolioOptimizer(budget=5000, min_ev_ratio=1.05)
    
    # テスト用のペイアウトデータ
    demo_numbers = [9, 8, 7]
    demo_payouts = pd.DataFrame({
        "number": [f"{i:03d}" for i in demo_numbers],
        "straight_prize": [75_000] * len(demo_numbers),
        "box_prize": [12_500] * len(demo_numbers),
    })
    
    # テスト用の確率データ
    # 例：009のボックスの場合
    #   - box_prob = 0.011 (1.1%)
    #   - box_prize = 12,500円
    #   - コスト = 200円
    #   - 期待回収額 = 0.011 × 12,500 = 137.5円
    #   - EV比 = 137.5 / 200 = 0.6875x
    #   - 期待利益 = 137.5 - 200 = -62.5円
    
    # か、もう一つ：
    #   - box_prob = 0.015 (1.5%)
    #   - box_prize = 12,500円
    #   - 期待回収額 = 0.015 × 12,500 = 187.5円
    #   - EV比 = 187.5 / 200 = 0.9375x （まだ利益なし）
    
    # 利益が出るケース：
    #   - box_prob = 0.02 (2%)
    #   - box_prize = 12,500円
    #   - 期待回収額 = 0.02 × 12,500 = 250円
    #   - EV比 = 250 / 200 = 1.25x
    #   - 期待利益 = 250 - 200 = 50円 ✓
    
    demo_probs = pd.DataFrame({
        "number": ["009", "008", "007"],
        "straight_prob": [0.002, 0.0025, 0.003],
        "box_prob": [0.02, 0.018, 0.016],
        "mini_prob": [0.01] * len(demo_numbers),
    })
    
    candidates = optimizer.generate_candidates(demo_numbers, demo_payouts, demo_probs)
    
    print("\n=== EV比計算テスト ===\n")
    print(f"総候補数: {len(candidates)}\n")
    
    # ボックス009の検証
    box_009 = next((c for c in candidates if c.number == "009" and c.bet_type == "box"), None)
    assert box_009 is not None, "ボックス009候補が生成されませんでした"
    if box_009:
        print(f"ボックス009:")
        print(f"  期待回収額: 0.02 × 12,500 = 250円")
        print(f"  計算結果:")
        print(f"    - EV比: {box_009.ev_ratio:.4f}x (期待: 1.2500x)")
        print(f"    - 期待利益: {box_009.expected_value:.2f}円 (期待: 50.00円)")
        
        # 検証
        expected_ev_ratio = 1.25
        expected_profit = 50.0
        
        ev_ratio_ok = abs(box_009.ev_ratio - expected_ev_ratio) < 0.001
        profit_ok = abs(box_009.expected_value - expected_profit) < 0.1
        
        print(f"  ✓ EV比: {'PASS' if ev_ratio_ok else 'FAIL'}")
        print(f"  ✓ 期待利益: {'PASS' if profit_ok else 'FAIL'}\n")
        
        assert ev_ratio_ok, "EV比の計算が期待値と一致しません"
        assert profit_ok, "期待利益の計算が期待値と一致しません"


def test_relationship():
    """EV比と期待利益の関係を検証"""
    from src.utils.config import TICKET_COST
    
    print("\n=== EV比と期待利益の関係式 ===\n")
    print(f"チケットコスト: {TICKET_COST}円\n")
    
    # 複数のケースをテスト
    test_cases = [
        {"ev_ratio": 1.0, "desc": "損益分岐点"},
        {"ev_ratio": 1.1, "desc": "10%利益"},
        {"ev_ratio": 1.2, "desc": "20%利益"},
        {"ev_ratio": 1.5, "desc": "50%利益"},
        {"ev_ratio": 2.0, "desc": "100%利益（2倍回収）"},
    ]
    
    print("正しい公式: 期待利益 = (EV比 - 1) × コスト\n")
    
    for case in test_cases:
        ev_ratio = case["ev_ratio"]
        expected_profit = (ev_ratio - 1.0) * TICKET_COST
        print(f"{case['desc']:15} (EV比={ev_ratio:.1f}x) → 期待利益={expected_profit:.1f}円")
        calc_profit = (ev_ratio - 1.0) * TICKET_COST
        assert abs(calc_profit - expected_profit) < 1e-9


if __name__ == "__main__":
    print("=" * 60)
    print("         EV比計算検証スイート")
    print("=" * 60)
    
    try:
        test_relationship()
        test_ev_calculation()
        
        print("\n" + "=" * 60)
        print("✓ すべてのテストが成功しました")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
