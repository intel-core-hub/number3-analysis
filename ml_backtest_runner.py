import argparse

import pandas as pd

from numbers3_logic import Numbers3MLBacktester


def main():
    parser = argparse.ArgumentParser(description="Run Numbers3 ML backtest")
    parser.add_argument("--csv", default="numbers3_clean.csv", help="Path to numbers3 data CSV")
    parser.add_argument("--window", type=int, default=300, help="Training window size")
    parser.add_argument("--rounds", type=int, default=50, help="Number of rounds to backtest")
    parser.add_argument("--valid-size", type=int, default=180, help="Validation size for tuning")
    parser.add_argument(
        "--tune-every",
        type=int,
        default=0,
        help="Tune hyperparams every N rounds (0 disables tuning)",
    )
    parser.add_argument("--output", default="ml_backtest_results.csv", help="Output CSV for results")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    tune_every = args.tune_every if args.tune_every > 0 else None

    backtester = Numbers3MLBacktester(
        df,
        window=args.window,
        test_rounds=args.rounds,
        valid_size=args.valid_size,
        tune_every=tune_every,
    )
    results, summary = backtester.run()
    results.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("ML backtest summary:")
    print(summary.to_string(index=False))
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
