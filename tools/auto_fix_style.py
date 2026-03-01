"""tools/auto_fix_style.py

簡単なプロジェクト内スタイル自動修正:
 - 行末の余分な空白を削除
 - ファイル末尾に改行を保証
 - 対象はワークスペース内の .py ファイル（.venv, venv を除外）

実行例:
  .venv\Scripts\python.exe tools/auto_fix_style.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".venv", "venv"}


def should_skip(p: Path) -> bool:
    for part in p.parts:
        if part in EXCLUDE_DIRS:
            return True
    return False


def fix_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return False

    # Normalize line endings to \n
    lines = text.splitlines()
    new_lines = [l.rstrip() for l in lines]

    # Ensure newline at EOF
    if len(new_lines) == 0:
        new_text = "\n"
    else:
        new_text = "\n".join(new_lines) + "\n"

    if new_text != text.replace("\r\n", "\n").replace("\r", "\n"):
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> None:
    py_files = list(ROOT.rglob("*.py"))
    modified = []
    for p in py_files:
        if should_skip(p):
            continue
        if fix_file(p):
            modified.append(str(p.relative_to(ROOT)))

    print(f"Fixed {len(modified)} files")
    for m in modified:
        print(" -", m)


if __name__ == "__main__":
    main()
