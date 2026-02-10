import re
import sys

import os

path = 'README.md'
if len(sys.argv) > 1:
    path = sys.argv[1]
if not os.path.exists(path):
    print(f'File not found: {path}')
    sys.exit(1)
with open(path, encoding='utf-8') as f:
    lines = f.read().splitlines()

issues = []

# Detect fenced code blocks and skip checks inside them
in_fence = False
fence_lang = None
in_fence_lines = [False] * (len(lines) + 1)
for i, line in enumerate(lines, start=1):
    m = re.match(r'^```\s*(.*)$', line)
    if m:
        lang = m.group(1)
        if not in_fence:
            # opening fence
            if lang.strip() == '':
                issues.append((i, 'MD040', 'Fenced code block opening missing language'))
            in_fence = True
            fence_lang = lang
            in_fence_lines[i] = True
        else:
            # closing fence
            in_fence_lines[i] = True
            in_fence = False
            fence_lang = None
        continue
    if in_fence:
        in_fence_lines[i] = True
        continue

# Headings blank line before (MD022)
for i, line in enumerate(lines, start=1):
    if in_fence_lines[i]:
        continue
    if re.match(r'^(#{1,6})\s+', line):
        if i > 1 and lines[i-2].strip() != '':
            issues.append((i, 'MD022', 'Heading should be preceded by a blank line'))

# Lists should be surrounded by blank lines (MD032) — detect list blocks
i = 1
while i <= len(lines):
    line = lines[i-1]
    if re.match(r'^\s*[-*+]\s+', line):
        # start of a list block
        start = i
        j = i
        while j <= len(lines) and re.match(r'^\s*[-*+]\s+', lines[j-1]):
            j += 1
        end = j - 1
        prev_blank = (start == 1) or (lines[start-2].strip() == '')
        prev_is_heading = (start > 1 and re.match(r'^(#{1,6})\s+', lines[start-2]))
        prev_is_numbered = (start > 1 and re.match(r'^\s*\d+\.\s+', lines[start-2]))
        prev_ends_with_colon = (start > 1 and lines[start-2].rstrip().endswith(':'))
        next_blank = (end == len(lines)) or (lines[end].strip() == '')
        if not (prev_blank or prev_is_heading or prev_is_numbered or prev_ends_with_colon):
            issues.append((start, 'MD032', 'List block should be preceded by a blank line'))
        if not next_blank:
            issues.append((end, 'MD032', 'List block should be followed by a blank line'))
        i = j
    else:
        i += 1

# Duplicate headings (MD024)
headings = {}
for i, line in enumerate(lines, start=1):
    m = re.match(r'^(#{1,6})\s+(.*)$', line)
    if m:
        text = m.group(2).strip()
        key = re.sub(r'`.*?`', '', text).strip().lower()
        headings.setdefault(key, []).append(i)
for k, locs in headings.items():
    if len(locs) > 1:
        issues.append((locs[0], 'MD024', f"Duplicate heading '{k}' at lines: {', '.join(map(str, locs))}"))

if issues:
    print('Found issues:')
    for line_no, code, msg in issues:
        print(f'{code} at line {line_no}: {msg}')
    sys.exit(2)
else:
    print('No issues found')
    sys.exit(0)
