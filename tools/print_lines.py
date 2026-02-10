import sys
path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    for i,l in enumerate(f,1):
        print(f'{i:03d}: {l.rstrip()}')
