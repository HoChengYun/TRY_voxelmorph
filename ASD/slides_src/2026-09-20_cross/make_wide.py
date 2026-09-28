# -*- coding: utf-8 -*-
"""mix_wide（U-Net 加寬 2 倍）那三頁要用的圖。

輸出到 models/deck_charts/：
    wide_unet.png        只畫 U-Net 那塊、字放大（手冊的 ASD/img/unet_explained.png 整張放進投影片字太小）
    wide_steps.png       三個改動各讓 Dice 進步多少（換版本 / 平滑權重砍半 / 加寬）
    wide_difficulty.png  每位受試者：起點 Dice vs 加寬多進步多少

數字從 deck_data.json 讀（gather.py 算的）；散布圖要逐人的點，直接讀兩份 dice csv。
"""
import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, 'ASD', 'img'))
from make_unet_explained import draw_unet  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(ROOT, 'models', 'deck_charts')
os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(HERE, 'deck_data.json'), encoding='utf-8'))
P, M, WD = D['paired'], D['models'], D['wide']

INK, MUTED, PAPER, RULE = '#141A1D', '#5F6A6B', '#FAFAF8', '#D9D9D2'
TEAL, TEAL_L, RUST = '#0E7C7B', '#5BB8B6', '#A34F1B'


def clean(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=.3)
    ax.set_axisbelow(True)


# ── 1. U-Net（投影片版，字放大）─────────────────────────────────────
fig = plt.figure(figsize=(11.6, 7.2), facecolor=PAPER)
draw_unet(fig.add_axes([0.005, 0.01, 0.9, 0.98]), fs=1.3, title=False)
fig.savefig(os.path.join(OUT, 'wide_unet.png'), dpi=130, facecolor=PAPER)
plt.close(fig)
print('->', os.path.join(OUT, 'wide_unet.png'))

# ── 2. 三個改動各貢獻多少 ────────────────────────────────────────────
steps = [('換成位移場版', 'mix_exp2 → mix_exp4', P['version_effect'], TEAL_L),
         ('平滑權重砍半', 'mix_exp4 → mix_exp3', P['lambda_effect'], TEAL_L),
         ('U-Net 加寬 2 倍', 'mix_exp3 → mix_wide', P['width_effect'], RUST)]
fig, ax = plt.subplots(figsize=(12, 3.4), facecolor=PAPER)
ax.set_facecolor(PAPER)
y = np.arange(len(steps))[::-1]
for yi, (name, how, st, col) in zip(y, steps):
    ax.barh(yi, st['mean'], height=0.58, color=col, zorder=3)
    ax.text(st['mean'] + 0.00012, yi, '+%.4f　（%d / %d 人變好）' % (st['mean'], st['win'], st['n']),
            va='center', fontsize=14, fontweight='bold', color=col if col == RUST else INK)
ax.set_yticks(y)
ax.set_yticklabels(['%s\n%s' % (n, h) for n, h, _, _ in steps], fontsize=13)
for lab, (_, _, _, col) in zip(ax.get_yticklabels(), steps):
    if col == RUST:
        lab.set_color(RUST)
        lab.set_fontweight('bold')
ax.set_xlim(0, 0.0085)
ax.set_xlabel('Dice 進步多少（同一批 51 位逐人相減的平均）', fontsize=12)
clean(ax)
ax.grid(axis='y', visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'wide_steps.png'), dpi=130, facecolor=PAPER)
plt.close(fig)
print('->', os.path.join(OUT, 'wide_steps.png'))


# ── 3. 越難的人幫越多 ───────────────────────────────────────────────
def dice_csv(p):
    with open(os.path.join(ROOT, p), encoding='utf-8') as f:
        return {r['file'][:-4]: float(r['dice_mean']) for r in csv.DictReader(f)}


B = dice_csv('models/mix_exp2/dice_baseline.csv')
E = dice_csv('models/mix_exp3/dice_%s.csv' % M['mix_exp3']['epoch'])
W = dice_csv('models/mix_wide/dice_%s.csv' % M['mix_wide']['epoch'])
K = sorted(set(B) & set(E) & set(W))
x = np.array([B[k] for k in K])
d = np.array([W[k] - E[k] for k in K])

fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=PAPER)
ax.set_facecolor(PAPER)
ax.axhline(0, color=MUTED, lw=1)
ax.scatter(x, d, s=55, color=TEAL, alpha=.75, edgecolor='white', linewidth=.8, zorder=3)
z = np.polyfit(x, d, 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, np.polyval(z, xs), color=RUST, lw=2.4, ls='--', zorder=4)
for s in ('VNT045', 'D031', 'T054', 'sub-0043', 'A0131'):
    i = K.index(s)
    ax.scatter(x[i], d[i], s=110, facecolor='none', edgecolor=INK, linewidth=1.6, zorder=5)
    ax.annotate(s, (x[i], d[i]), (8, 7), textcoords='offset points', fontsize=11.5, color=INK)
ax.set_xlabel('起點 Dice（只做線性對位、還沒用模型）　← 越左越難對', fontsize=12.5)
ax.set_ylabel('加寬後多進步多少', fontsize=12.5)
ax.text(.02, .93, '起點最差 10 位：平均 +%.4f' % WD['hard10'], transform=ax.transAxes,
        fontsize=14, fontweight='bold', color=RUST, va='top',
        bbox=dict(fc=PAPER, ec='none', pad=3), zorder=6)
ax.text(.98, .93, '起點最好 10 位：平均 +%.4f' % WD['easy10'], transform=ax.transAxes,
        fontsize=14, fontweight='bold', color=TEAL, va='top', ha='right',
        bbox=dict(fc=PAPER, ec='none', pad=3), zorder=6)
ax.text(.98, .20, '相關 %.2f' % WD['r_base'], transform=ax.transAxes,
        fontsize=13, color=RUST, ha='right', bbox=dict(fc=PAPER, ec='none', pad=3), zorder=6)
clean(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'wide_difficulty.png'), dpi=130, facecolor=PAPER)
plt.close(fig)
print('->', os.path.join(OUT, 'wide_difficulty.png'))
