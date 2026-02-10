import re
import sys


def check_file(path):
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    file_issues = []

    # Detect fenced code blocks (only consider opening fences)
    fence_lines = set()
    in_fence = False
    for i, line in enumerate(lines, start=1):
        m = re.match(r'^```\s*(.*)$', line)
        if m:
            lang = m.group(1)
            fence_lines.add(i)
            if not in_fence:
                if lang.strip() == '':
                    file_issues.append((i, 'MD040', 'Fenced code block opening missing language'))
                in_fence = True
            else:
                in_fence = False
        elif in_fence:
            fence_lines.add(i)

    # Headings blank line before (MD022) — ignore inside code fences
    for i, line in enumerate(lines, start=1):
        if i in fence_lines:
            continue
        if re.match(r'^(#{1,6})\s+', line):
            if i > 1 and lines[i - 2].strip() != '':
                file_issues.append((i, 'MD022', 'Heading should be preceded by a blank line'))

    # Lists should be surrounded by blank lines (MD032) — check top-level list blocks only
    i = 1
    while i <= len(lines):
        if i in fence_lines:
            i += 1
            continue
        line = lines[i - 1]
        is_bullet = re.match(r'^[-*+]\s+', line)
        is_ordered = re.match(r'^\d+\.\s+', line)
        if is_bullet or is_ordered:
            start = i
            j = i
            while j <= len(lines):
                if j in fence_lines:
                    break
                next_line = lines[j - 1]
                if next_line.strip() == '':
                    break
                is_list_item = re.match(r'^[-*+]\s+', next_line) or re.match(r'^\d+\.\s+', next_line)
                is_continuation = next_line.startswith(' ') or next_line.startswith('\t')
                if is_list_item or is_continuation:
                    j += 1
                else:
                    break
            end = j - 1
            prev_blank = (start == 1) or (lines[start - 2].strip() == '')
            prev_is_heading = (start > 1 and re.match(r'^(#{1,6})\s+', lines[start - 2]))
            next_blank = (end == len(lines)) or (lines[end].strip() == '')
            if not (prev_blank or prev_is_heading):
                file_issues.append((start, 'MD032', 'List block should be preceded by a blank line'))
            if not next_blank:
                file_issues.append((end, 'MD032', 'List block should be followed by a blank line'))
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
    for key, locs in headings.items():
        if len(locs) > 1:
            file_issues.append(
                (locs[0], 'MD024', f"Duplicate heading '{key}' at lines: {', '.join(map(str, locs))}")
            )

    return file_issues


def main():
    paths = sys.argv[1:] or ['README.md']
    issues = []

    for path in paths:
        issues.extend([(path, *issue) for issue in check_file(path)])

    if issues:
        print('Found issues:')
        for path, line_no, code, msg in issues:
            print(f'{path}: {code} at line {line_no}: {msg}')
        sys.exit(2)

    print('No issues found')
    sys.exit(0)


if __name__ == '__main__':
    main()
