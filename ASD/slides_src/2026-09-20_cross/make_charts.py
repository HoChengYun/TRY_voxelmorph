# -*- coding: utf-8 -*-
"""產生這份簡報要用的兩張總覽圖（其餘圖直接引用 models/ 底下既有的輸出）。

輸出到 models/deck_charts/：
    overview_four_models.png   四顆模型：起點 -> 配準後
    contribution.png           模型貢獻（配準後 減 起點），含 95% 信賴區間
    cross_eval.png             交叉測試：模型用哪套影像訓練 × 拿哪套影像測試

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


# ── 圖三：交叉測試 ────────────────────────────────────────────────────
X = 'models/mix_exp2/cross_mix_tiger_exp2_exp3/'
cross = {
    'tiger_on_fs_v2': rd(X + 'tiger_exp2_on_freesurfer.csv'),
    'tiger_on_fs_v3': rd(X + 'tiger_exp3_on_freesurfer.csv'),
    'mix_on_tg_v2': rd(X + 'mix_exp2_on_tigerbx.csv'),
    'mix_on_tg_v3': rd(X + 'mix_exp3_on_tigerbx.csv'),
}
mean = lambda d: float(np.mean([d[k] for k in K]))
panels = [
    ('測試：FreeSurfer 的影像', np.mean([base_fs[k] for k in K]), [
        ('用 FreeSurfer\n影像訓練', mean(M[0][2]), C['teal']),
        ('用 tigerbx\n影像訓練', mean(cross['tiger_on_fs_v2']), C['rust']),
        ('用 FreeSurfer\n影像訓練', mean(M[1][2]), C['teal']),
        ('用 tigerbx\n影像訓練', mean(cross['tiger_on_fs_v3']), C['rust']),
    ]),
    ('測試：tigerbx 的影像', np.mean([base_tg[k] for k in K]), [
        ('用 FreeSurfer\n影像訓練', mean(cross['mix_on_tg_v2']), C['teal']),
        ('用 tigerbx\n影像訓練', mean(M[2][2]), C['rust']),
        ('用 FreeSurfer\n影像訓練', mean(cross['mix_on_tg_v3']), C['teal']),
        ('用 tigerbx\n影像訓練', mean(M[3][2]), C['rust']),
    ]),
]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for ax, (title, b0, bars) in zip(axes, panels):
    for i, (lab, v, col) in enumerate(bars):
        x = i + (0.25 if i >= 2 else 0)
        ax.bar(x, v, color=col, width=.72)
        ax.text(x, v + .004, '%.3f' % v, ha='center', fontsize=11, fontweight='bold')
    ax.axhline(b0, ls='--', color='#C0392B', lw=1.3)
    ax.text(-0.42, b0 + .004, '沒用模型、只做線性對位 %.3f' % b0, ha='left', color='#C0392B',
            fontsize=9.5, bbox=dict(fc='white', ec='none', pad=1.5))
    ax.set_xticks([0, 1, 2.25, 3.25])
    ax.set_xticklabels([b[0] for b in bars], fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0.66, 0.90)
    ax.grid(axis='y', alpha=.3)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for xx, tt in ((0.5, '速度場版'), (2.75, '位移場版')):
        ax.text(xx, 0.888, tt, ha='center', va='top', fontsize=12, fontweight='bold',
                color=C['ink'], bbox=dict(fc='#EFEFEB', ec=C['rule'], pad=3))
axes[0].set_ylabel('Dice（test 51 位）', fontsize=11)
fig.suptitle('每一格裡，兩根柱子測的是同一批影像，只是模型訓練時看的影像不同',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'cross_eval.png'), dpi=130)
print('->', os.path.join(OUT, 'cross_eval.png'))
