# -*- coding: utf-8 -*-
"""產生這份簡報要用的兩張總覽圖（其餘圖直接引用 models/ 底下既有的輸出）。

輸出到 models/deck_charts/：
    overview_four_models.png   四顆模型：起點 -> 配準後
    contribution.png           模型貢獻（配準後 減 起點），含 95% 信賴區間

數字全部從 models/*/dice_*.csv 讀，不手打。
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
os.makedirs(OUT, exist_ok=True)

C = {'ink': '#141A1D', 'muted': '#5F6A6B', 'rule': '#D9D9D2',
     'teal': '#0E7C7B', 'teal_l': '#5BB8B6', 'rust': '#A34F1B', 'rust_l': '#D9895A'}


def rd(p):
    with open(os.path.join(ROOT, p), encoding='utf-8') as f:
        return {r['file'][:-4]: float(r['dice_mean']) for r in csv.DictReader(f)}


base_fs = rd('models/mix_exp2/dice_baseline.csv')
base_tg = rd('models/tiger_exp2/dice_baseline.csv')
M = [
    ('mix_exp2',   'FreeSurfer 標籤\n速度場版', rd('models/mix_exp2/dice_0240.csv'),   base_fs, C['teal']),
    ('mix_exp3',   'FreeSurfer 標籤\n位移場版', rd('models/mix_exp3/dice_0240.csv'),   base_fs, C['teal_l']),
    ('tiger_exp2', 'tigerbx 標籤\n速度場版',   rd('models/tiger_exp2/dice_0250.csv'), base_tg, C['rust']),
    ('tiger_exp3', 'tigerbx 標籤\n位移場版',   rd('models/tiger_exp3/dice_0210.csv'), base_tg, C['rust_l']),
]
K = sorted(base_fs)

# ── 圖一：起點 -> 配準後 ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.2))
x = np.arange(len(M))
for i, (name, lab, after, base, col) in enumerate(M):
    b = np.mean([base[k] for k in K])
    a = np.mean([after[k] for k in K])
    ax.bar(i - 0.17, b, width=0.32, color=C['rule'], edgecolor=C['muted'], linewidth=.6)
    ax.bar(i + 0.17, a, width=0.32, color=col)
    ax.text(i - 0.17, b + .004, '%.3f' % b, ha='center', fontsize=10, color=C['muted'])
    ax.text(i + 0.17, a + .004, '%.3f' % a, ha='center', fontsize=12, fontweight='bold')
    ax.annotate('', xy=(i + .17, a - .004), xytext=(i - .17, b + .012),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.6))
ax.set_xticks(x)
ax.set_xticklabels([m[1] for m in M], fontsize=11)
ax.set_ylim(0.60, 0.92)
ax.set_ylabel('Dice（30 個結構，test 51 位）', fontsize=11)
ax.set_title('灰色＝只做線性對位的起點　　彩色＝加上非線性配準之後', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=.3)
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'overview_four_models.png'), dpi=130)
print('->', os.path.join(OUT, 'overview_four_models.png'))

# ── 圖二：模型貢獻 ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.6))
for i, (name, lab, after, base, col) in enumerate(M):
    g = np.array([after[k] - base[k] for k in K])
    ax.barh(i, g.mean(), color=col, height=.6,
            xerr=1.96 * g.std(ddof=1) / len(g) ** .5, capsize=4, error_kw={'ecolor': C['ink']})
    ax.text(g.mean() + .009, i, '+%.3f' % g.mean(), va='center', fontsize=13, fontweight='bold')
ax.set_yticks(range(len(M)))
ax.set_yticklabels([m[1].replace('\n', ' ') for m in M], fontsize=11)
ax.invert_yaxis()
ax.set_xlim(0, 0.16)
ax.set_xlabel('配準後 減 起點（Dice 進步了多少）', fontsize=11)
ax.grid(axis='x', alpha=.3)
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'contribution.png'), dpi=130)
print('->', os.path.join(OUT, 'contribution.png'))
