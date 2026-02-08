# Numbers3 Update Automation

## Auto update script

Run the update script from the project root:

```bash
python auto_update.py
```

## GitHub Actions

The workflow `.github/workflows/update_data.yml` runs on a schedule and can be triggered manually.
It installs dependencies from `requirements.txt`, runs the update script, and commits `numbers3_clean.csv`
when changes are detected.

## ML backtest

Run the ML backtest runner from the project root:

```bash
python ml_backtest_runner.py --rounds 50 --window 300
```

Enable periodic hyperparameter tuning (slower):

```bash
python ml_backtest_runner.py --rounds 30 --tune-every 5 --valid-size 120
```
