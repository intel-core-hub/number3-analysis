import os

# 設定
OUTPUT_FILE = "repomix-output.xml"
IGNORE_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".vscode", 
    "node_modules", ".ipynb_checkpoints", "dist", "build", "site-packages"
}
IGNORE_EXTENSIONS = {
    ".pyc", ".exe", ".dll", ".so", ".bak", ".ttf", ".woff", ".woff2", 
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".tar", ".gz", ".7z",
    ".pdf"
}
IGNORE_FILES = {
    OUTPUT_FILE, 
    os.path.basename(__file__), # このスクリプト自体
    "package-lock.json", "yarn.lock", "poetry.lock"
}

def is_text_file(filepath):
    """ファイルがテキストファイルかどうかを簡易判定する"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read(4096) # 先頭4KBを読んでみる
        return True
    except UnicodeDecodeError:
        return False
    except Exception:
        return False

def generate_repomix_output():
    root_dir = os.getcwd()
    output_path = os.path.join(root_dir, OUTPUT_FILE)
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write("<repository>\n")
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 除外ディレクトリをリストから削除（in-placeで変更することでos.walkがその中に入らない）
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            
            for filename in filenames:
                if filename in IGNORE_FILES:
                    continue
                
                _, ext = os.path.splitext(filename)
                if ext.lower() in IGNORE_EXTENSIONS:
                    continue
                
                filepath = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(filepath, root_dir)
                
                # 出力ファイル自体はスキップ（念のため）
                if os.path.abspath(filepath) == os.path.abspath(output_path):
                    continue

                if is_text_file(filepath):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            
                        outfile.write(f'<file path="{rel_path}">\n')
                        outfile.write(content)
                        # ファイル末尾が改行でない場合は改行を追加
                        if content and not content.endswith('\n'):
                            outfile.write('\n')
                        outfile.write("</file>\n")
                        print(f"Added: {rel_path}")
                    except Exception as e:
                        print(f"Skipped (error reading): {rel_path} - {e}")
                else:
                    print(f"Skipped (binary): {rel_path}")

        outfile.write("</repository>\n")
    
    print(f"\nSuccessfully generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_repomix_output()
