"""
曜日特徴の予測価値を詳細に分析する。

出力:
- results/weekday_counts.csv
- results/weekday_digit_position_stats.csv
- results/weekday_mi_summary.csv
- results/weekday_js_divergence.csv
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np
import pandas as pd

print('=' * 70)
print('曜日特徴の詳細分析')
print('=' * 70)

# データ読込
df = pd.read_csv('numbers3_clean.csv', encoding='utf-8-sig')
df['dt'] = pd.to_datetime(df['抽せん日'], errors='coerce')

# 当選番号から桁を作成
num = df['当選番号'].astype(str).str.zfill(3)
df['n1'] = num.str[0].astype(int)
df['n2'] = num.str[1].astype(int)
df['n3'] = num.str[2].astype(int)

# 基本列
df['weekday'] = df['dt'].dt.weekday
df['weekday_name'] = df['weekday'].map({0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
df['digit_sum'] = df[['n1', 'n2', 'n3']].sum(axis=1)

# 分割: 2004-07-01
cutoff = pd.Timestamp('2004-07-01')
periods = {
    'all': df,
    'pre_2004_07': df[df['dt'] < cutoff],
    'post_2004_07': df[df['dt'] >= cutoff],
}

print(f'\n[INFO] 総データ数: {len(df)}')
print(f'[INFO] 期間: {df["dt"].min()} - {df["dt"].max()}')
print(f'[INFO] 旧システム比率: {len(periods["pre_2004_07"]) / len(df) * 100:.1f}%')

# 基本集計
weekday_counts = (
    df.groupby(['weekday', 'weekday_name'])
      .size()
      .reset_index(name='count')
      .sort_values('weekday')
)
weekday_counts['pct'] = weekday_counts['count'] / weekday_counts['count'].sum() * 100
weekday_counts.to_csv('results/weekday_counts.csv', index=False, encoding='utf-8-sig')

# ユーティリティ

def entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(entropy(m) - 0.5 * (entropy(p) + entropy(q)))


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    # x, y are integer labels
    x = x.astype(int)
    y = y.astype(int)
    x_vals = np.unique(x)
    y_vals = np.unique(y)
    px = np.array([np.mean(x == xv) for xv in x_vals])
    py = np.array([np.mean(y == yv) for yv in y_vals])
    mi = 0.0
    for i, xv in enumerate(x_vals):
        for j, yv in enumerate(y_vals):
            pxy = np.mean((x == xv) & (y == yv))
            if pxy <= 0:
                continue
            mi += pxy * math.log2(pxy / (px[i] * py[j]))
    return float(mi)


def cramers_v(table: np.ndarray) -> float:
    # Cramer's V = sqrt(chi2 / (n * (k-1)))
    n = table.sum()
    if n == 0:
        return 0.0
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((table - expected) ** 2 / expected)
    r, c = table.shape
    k = min(r, c)
    if k <= 1:
        return 0.0
    return float(math.sqrt(chi2 / (n * (k - 1))))


# 1) 曜日 x 桁 (n1/n2/n3) の効果量
position_stats = []
for period_name, pdf in periods.items():
    pdf = pdf.dropna(subset=['weekday', 'n1', 'n2', 'n3'])
    for pos in ['n1', 'n2', 'n3']:
        table = pd.crosstab(pdf['weekday'], pdf[pos]).reindex(index=range(7), fill_value=0).values
        v = cramers_v(table)
        mi = mutual_information(pdf['weekday'].values, pdf[pos].values)
        position_stats.append({
            'period': period_name,
            'position': pos,
            'cramers_v': v,
            'mutual_information': mi,
            'samples': len(pdf)
        })

position_stats_df = pd.DataFrame(position_stats)
position_stats_df.to_csv('results/weekday_digit_position_stats.csv', index=False, encoding='utf-8-sig')

# 2) digit_sum との関連 (MI)
mi_summary = []
for period_name, pdf in periods.items():
    pdf = pdf.dropna(subset=['weekday', 'digit_sum'])
    mi_sum = mutual_information(pdf['weekday'].values, pdf['digit_sum'].values)
    mi_summary.append({
        'period': period_name,
        'feature': 'digit_sum',
        'mutual_information': mi_sum,
        'samples': len(pdf)
    })

# 3) weekday のみでの情報量 (エントロピー)
for period_name, pdf in periods.items():
    counts = pdf['weekday'].value_counts().sort_index().reindex(range(7), fill_value=0).values
    probs = counts / counts.sum()
    mi_summary.append({
        'period': period_name,
        'feature': 'weekday_entropy',
        'mutual_information': entropy(probs),
        'samples': len(pdf)
    })

mi_summary_df = pd.DataFrame(mi_summary)
mi_summary_df.to_csv('results/weekday_mi_summary.csv', index=False, encoding='utf-8-sig')

# 4) JS divergence: 曜日別の digit 分布の差
js_rows = []
for period_name, pdf in periods.items():
    pdf = pdf.dropna(subset=['weekday', 'n1', 'n2', 'n3'])
    # 全体分布 (3桁合算)
    all_digits = pd.concat([pdf['n1'], pdf['n2'], pdf['n3']])
    overall = all_digits.value_counts().reindex(range(10), fill_value=0).values.astype(float)

    for wd in range(7):
        subset = pdf[pdf['weekday'] == wd]
        if len(subset) == 0:
            continue
        digits = pd.concat([subset['n1'], subset['n2'], subset['n3']])
        dist = digits.value_counts().reindex(range(10), fill_value=0).values.astype(float)
        js = js_divergence(dist, overall)
        js_rows.append({
            'period': period_name,
            'weekday': wd,
            'weekday_name': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][wd],
            'js_divergence': js,
            'samples': len(subset)
        })

js_df = pd.DataFrame(js_rows)
js_df.to_csv('results/weekday_js_divergence.csv', index=False, encoding='utf-8-sig')

# サマリ出力
print('\n[INFO] 保存ファイル:')
print('  results/weekday_counts.csv')
print('  results/weekday_digit_position_stats.csv')
print('  results/weekday_mi_summary.csv')
print('  results/weekday_js_divergence.csv')

print('\n[INFO] 主要指標 (Cramer\'s V, MI):')
print(position_stats_df.sort_values(['period', 'position']).to_string(index=False))

print('\n[INFO] MI summary:')
print(mi_summary_df.sort_values(['period', 'feature']).to_string(index=False))

print('\n[INFO] JS divergence (weekday vs overall digit distribution):')
print(js_df.sort_values(['period', 'js_divergence'], ascending=[True, False]).head(12).to_string(index=False))

print('\n' + '=' * 70)
print('[INFO] 分析完了')
print('=' * 70)
