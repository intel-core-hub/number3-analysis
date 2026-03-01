import sys
from pathlib import Path


def main(path: str, limit: int = 120) -> None:
    p = Path(path)
    if not p.exists():
        print(f"File not found: {path}")
        sys.exit(2)
    for i, l in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if len(l) > limit:
            print(i, len(l), l)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: find_long_lines.py <file> [limit]")
        sys.exit(2)
    path = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    main(path, limit)
