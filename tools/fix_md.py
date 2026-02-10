import sys
import re
import os

def guess_language(next_line):
    s = next_line.strip()
    if not s:
        return 'text'
    if s.startswith('#') or s.startswith('import ') or s.startswith('def ') or s.startswith('class ') or s.startswith('for ') or s.startswith('with ') or s.startswith('from '):
        return 'python'
    if s.startswith('python ') or s.startswith('streamlit ') or s.startswith('pip ') or s.startswith('$'):
        return 'bash'
    return 'text'


def fix_file(path):
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # add blank line before headings if needed
        if re.match(r'^(#{1,6})\s+', line):
            if out and out[-1].strip() != '':
                out.append('')
            out.append(line)
            i += 1
            continue
        # handle list blocks
        if re.match(r'^\s*[-*+]\s+', line):
            # ensure blank line before if not heading
            if out and out[-1].strip() != '' and not re.match(r'^(#{1,6})\s+', out[-1]):
                out.append('')
            # copy list block
            while i < n and re.match(r'^\s*[-*+]\s+', lines[i]):
                out.append(lines[i])
                i += 1
            # ensure blank line after
            if i < n and lines[i].strip() != '':
                out.append('')
            continue
        # handle fenced code block opening without language
        m = re.match(r'^```\s*$', line)
        if m:
            # lookahead to guess language
            j = i + 1
            next_line = ''
            while j < n and lines[j].strip() == '':
                j += 1
            if j < n:
                next_line = lines[j]
            lang = guess_language(next_line)
            out.append(f'```{lang}')
            i += 1
            # copy until closing fence
            while i < n and not re.match(r'^```\s*$', lines[i]):
                out.append(lines[i])
                i += 1
            # append closing fence if exists
            if i < n and re.match(r'^```\s*$', lines[i]):
                out.append('```')
                i += 1
            continue
        out.append(line)
        i += 1

    # write back only if changed
    new_content = '\n'.join(out) + ('\n' if out and out[-1] != '' else '')
    orig = '\n'.join(lines) + ('\n' if lines and lines[-1] != '' else '')
    if new_content != orig:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Fixed: {path}')
    else:
        print(f'No changes: {path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: fix_md.py <file.md>')
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print('Not found:', path)
        sys.exit(1)
    fix_file(path)
