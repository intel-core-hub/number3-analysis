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
