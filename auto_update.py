import logging
import sys

from numbers3_logic import update_numbers3_clean, validate_numbers3


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> int:
    logging.info("Numbers3 data update starting...")
    try:
        df = update_numbers3_clean(
            clean_path="numbers3_clean.csv",
            backup=True,
            sleep_seconds=1.0,
            force_full=False,
        )
        errors = validate_numbers3(df, strict=False, allow_missing_rounds=True)
        if not errors:
            logging.info("Update finished successfully: %s rows", len(df))
        else:
            logging.warning("Update finished with warnings: %s", errors)
        return 0
    except Exception:
        logging.exception("Update failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
