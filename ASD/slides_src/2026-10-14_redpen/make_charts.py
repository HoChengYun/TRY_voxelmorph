# -*- coding: utf-8 -*-
"""10/14 簡報的圖 -> models/deck_charts/1014_*.png。數字讀 deck_data.json（先跑 gather.py）。

  1014_folding_regions.png  ① 擠爆的點落在哪些區域（只留「佔幾 %」那一格、字放大）
  1014_lambda.png           ② 平滑權重 2 / 1 / 0.5：速度場 vs 位移場（還沒跑完的點標「跑中」）
  1014_dilution.png         ③ 30 個結構一起平均 vs 只平均殘留旁邊的結構（170 人）
  1014_regions.png          ③④ 頭頂／顱底／後腦杓：殘留多寡跟模型貢獻的相關
  1014_top_example.png      ③ 頭頂放大：殘留多的一位 vs 乾淨的一位（只放加標記的那張、字放大）
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, 'ASD'))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
D = json.load(open(os.path.join(HERE, 'deck_data.json'), encoding='utf-8'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
INK, MUTED, PAPER, RULE = '#141A1D', '#5F6A6B', '#FAFAF8', '#D9D9D2'
TEAL, RUST, RED = '#0E7C7B', '#A34F1B', '#C0392B'


def clean(ax):
    ax.grid(alpha=.3)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


# ── ② 平滑權重 ─────────────────────────────────────────────────────────
M = D['models']
W = [2.0, 1.0, 0.5]                      # 由左到右越放鬆
SERIES = [('速度場（全尺寸）', TEAL, {2.0: 'mix_exp5', 1.0: 'mix_exp6', 0.5: 'mix_exp7'}),
          ('位移場（全尺寸）', RUST, {2.0: 'mix_exp4', 1.0: 'mix_exp3'})]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), facecolor=PAPER)
jtxt = lambda y: '0%' if y == 0 else ('< 0.001%' if y < 0.001 else '%.3f%%' % y)   # 速度場權重 1、0.5 只有零星幾點
for ax, key, title in ((axes[0], 'mean', '對得多準（test Dice）'), (axes[1], 'jneg', '有沒有擠爆（擠爆點的比例 %）')):
    pts = {name: {i: M[exps[w]][key] for i, w in enumerate(W) if w in exps and M[exps[w]]['status'] == 'done'}
           for name, col, exps in SERIES}
    for name, col, exps in SERIES:
        xs = sorted(pts[name])
        ys = [pts[name][i] for i in xs]
        ax.plot(xs, ys, color=col, lw=2.6, marker='o', ms=9, label=name, zorder=3)
        for x, y in zip(xs, ys):
            if key == 'mean':            # 兩條線靠很近：同一個位置比高低，高的數字放上面、低的放下面
                other = [pts[n][x] for n in pts if n != name and x in pts[n]]
                up = not other or y >= max(other)
                ax.annotate('%.3f' % y, (x, y), textcoords='offset points', xytext=(0, 11 if up else -13),
                            ha='center', va='bottom' if up else 'top', fontsize=12, fontweight='bold', color=col)
            else:
                ax.annotate(jtxt(y), (x, y), textcoords='offset points', xytext=(0, 11), ha='center',
                            fontsize=12, fontweight='bold', color=col)
        for i, w in enumerate(W):
            if w in exps and M[exps[w]]['status'] == 'pending':
                ax.annotate('%s\n跑中' % exps[w], (i, 0), xycoords=('data', 'axes fraction'), xytext=(0, 34),
                            textcoords='offset points', ha='center', fontsize=11, color=col,
                            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=col, ls='--', lw=1.2))
    ax.set_xticks(range(len(W)))
    ax.set_xticklabels(['平滑權重 %g' % w for w in W], fontsize=12)
    ax.set_xlim(-0.5, len(W) - 0.5)
    ax.set_title(title, fontsize=13.5, fontweight='bold')
    ax.annotate('', xy=(0.98, -0.16), xytext=(0.02, -0.16), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.2))
    ax.text(0.5, -0.22, '越往右，平滑的限制越鬆（形變可以捏得越用力）', transform=ax.transAxes, ha='center',
            fontsize=11, color=MUTED)
    clean(ax)
axes[0].set_ylim(0.795, 0.812)
axes[0].legend(fontsize=11.5, frameon=False, loc='upper left')
axes[1].set_ylim(0, 0.42)
axes[1].axhline(0.366, color=RED, ls='--', lw=1.2)
axes[1].text(len(W) - 0.55, 0.373, '論文的位移場版 0.366%', ha='right', va='bottom', fontsize=11, color=RED)
fig.tight_layout(rect=[0, 0.04, 1, 1])
save(fig, '1014_lambda.png')

# ── ② 平滑權重變小，哪些結構變好、哪些變差（跟權重 2 比）──────────────────────────
S = D.get('struct', {})
if all(e in S for e in ('mix_exp5', 'mix_exp6', 'mix_exp7')):
    names = sorted(S['mix_exp5'], key=lambda n: S['mix_exp7'][n] - S['mix_exp5'][n])
    d1 = np.array([S['mix_exp6'][n] - S['mix_exp5'][n] for n in names])
    d05 = np.array([S['mix_exp7'][n] - S['mix_exp5'][n] for n in names])
    fig, ax = plt.subplots(figsize=(9.2, 7.0), facecolor=PAPER)    # 簡報上放左半邊，圖小一點、字才不會縮太小
    y = np.arange(len(names))
    ax.barh(y + 0.2, d1, height=0.38, color='#5BB8B6', label='權重 1（mix_exp6）')
    ax.barh(y - 0.2, d05, height=0.38, color='#0A4F4E', label='權重 0.5（mix_exp7）')
    for yy, v in zip(y - 0.2, d05):
        ax.text(v + (0.0012 if v >= 0 else -0.0012), yy, '%+.3f' % v, va='center', ha='left' if v >= 0 else 'right',
                fontsize=12, color='#0A4F4E', fontweight='bold')
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=13.5)
    ax.tick_params(axis='x', labelsize=11.5)
    lim = max(abs(d05).max(), abs(d1).max()) + 0.014
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('跟平滑權重 2（mix_exp5）比，Dice 變多少（左右平均）', fontsize=13)
    ax.legend(fontsize=13, frameon=False, loc='lower right')
    clean(ax)
    ax.grid(axis='y', alpha=0)
    fig.tight_layout()
    save(fig, '1014_lambda_struct.png')

# ── ③ 30 個結構一起平均 vs 只平均殘留旁邊的結構 ─────────────────────────────
from skullstrip_label_dice_pooled import load, COHORTS
rows = load()
ctx = lambda row: float(np.mean([float(row['label_%d' % l]) for l in (3, 42)]))
for r in rows:
    r['x'] = float(r['res']['top_vertex_mm'])
    r['g_all'] = float(r['after']['dice_mean']) - float(r['before']['dice_mean'])
    r['g_ctx'] = ctx(r['after']) - ctx(r['before'])
for cn in [c[0] for c in COHORTS]:                     # 批內百分位、批內置中（同 gather.py）
    rr = [r for r in rows if r['cohort'] == cn]
    xs = np.array([r['x'] for r in rr])
    ma, mc = np.mean([r['g_all'] for r in rr]), np.mean([r['g_ctx'] for r in rr])
    for r in rr:
        r['pct'] = (np.sum(xs < r['x']) + 0.5 * (np.sum(xs == r['x']) - 1)) / (len(xs) - 1)
        r['gc_all'], r['gc_ctx'] = r['g_all'] - ma, r['g_ctx'] - mc
DL = D['dilution']
ptxt = lambda p: 'p < 0.001' if p < 0.001 else 'p = %.2f' % p
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), facecolor=PAPER, sharey=True)
for ax, key, title, rr, pp, col in ((axes[0], 'gc_all', '以前的算法：30 個結構一起平均', DL['r_all'], DL['p_all'], MUTED),
                                    (axes[1], 'gc_ctx', '老師的算法：只平均殘留旁邊的結構（左右大腦皮質）',
                                     DL['r_ctx'], DL['p_ctx'], RUST)):
    x = np.array([r['pct'] for r in rows])
    y = np.array([r[key] for r in rows])
    ax.scatter(x, y, s=24, alpha=0.7, color=col, edgecolors='none')
    b1, b0 = np.polyfit(x, y, 1)
    ax.plot([0, 1], [b0, b0 + b1], color=INK, lw=2)
    ax.axhline(0, color=MUTED, lw=0.8, ls=':')
    ax.set_title('%s\nr = %+.2f，%s' % (title, rr, ptxt(pp)), fontsize=13, fontweight='bold')
    ax.set_xlabel('頭頂殘留多寡（左 = 最少，右 = 最多）', fontsize=11.5)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['最少', '中間', '最多'], fontsize=11)
    clean(ax)
axes[0].set_ylabel('模型貢獻（跟同一批的平均比）', fontsize=11.5)
fig.tight_layout()
save(fig, '1014_dilution.png')

# ── ③④ 三個位置 ───────────────────────────────────────────────────────
R = D['residue']
names = [('top', '頭頂'), ('base', '顱底'), ('back', '後腦杓')]
fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=PAPER)
for i, (k, n) in enumerate(names):
    v, p = R[k]['pooled_r'], R[k]['pooled_p']
    col = RUST if p < 0.05 else '#B9BDB9'
    ax.barh(i, v, height=0.55, color=col)
    ax.text(v - 0.012 if v < 0 else v + 0.012, i, 'r = %+.2f（%s）' % (v, ptxt(p)), va='center',
            ha='right' if v < 0 else 'left', fontsize=12.5, fontweight='bold', color=INK)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n for _, n in names], fontsize=13)
ax.invert_yaxis()
ax.axvline(0, color=INK, lw=1)
ax.set_xlim(-0.75, 0.35)
ax.set_xlabel('殘留越多，模型貢獻越……（負 = 越少；%d 人）' % R['top']['pooled_n'], fontsize=11.5)
clean(ax)
ax.grid(axis='y', alpha=0)
fig.tight_layout()
save(fig, '1014_regions.png')

# ── ① 擠爆的點落在哪些區域（佔幾 %）─────────────────────────────────────────
FOLD = D['folding']
REG = ['大腦皮質', '大腦白質', '腦內、沒有標籤', '腦室・腦脊髓液・脈絡叢', '小腦・腦幹', '深部灰質・海馬・杏仁核', '腦外（背景）']
SHOW = {'大腦皮質': '大腦皮質', '大腦白質': '大腦白質', '腦內、沒有標籤': '腦內沒有標籤（多是腦溝空隙）',
        '腦室・腦脊髓液・脈絡叢': '腦室・腦脊髓液', '小腦・腦幹': '小腦・腦幹',
        '深部灰質・海馬・杏仁核': '深部灰質・海馬・杏仁核', '腦外（背景）': '腦外'}
FCOL = {'mix_exp4': '#D9895A', 'mix_exp3': '#A34F1B', 'mix_wide': '#6B3FA0'}
FLAB = {'mix_exp4': '位移場・平滑權重 2', 'mix_exp3': '位移場・平滑權重 1', 'mix_wide': '位移場・平滑權重 1・加寬'}
fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=PAPER)
wbar = 0.26
for j, e in enumerate(('mix_exp4', 'mix_exp3', 'mix_wide')):
    v = [FOLD[e]['share'][r] for r in REG]
    ax.barh(np.arange(len(REG)) + (j - 1) * wbar, v, height=wbar * 0.92, color=FCOL[e], label=FLAB[e])
for i, r in enumerate(REG):
    ax.text(max(FOLD[e]['share'][r] for e in ('mix_exp4', 'mix_exp3', 'mix_wide')) + 0.8, i, '%.1f%%' % FOLD['mix_exp3']['share'][r],
            va='center', fontsize=12.5, fontweight='bold', color=FCOL['mix_exp3'])
ax.set_yticks(range(len(REG)))
ax.set_yticklabels([SHOW[r] for r in REG], fontsize=13)
ax.invert_yaxis()
ax.set_xlabel('擠爆的點有幾 % 落在這一區（數字是平滑權重 1 那顆）', fontsize=12)
ax.legend(fontsize=12, frameon=False, loc='lower right')
clean(ax)
ax.grid(axis='y', alpha=0)
fig.tight_layout()
save(fig, '1014_folding_regions.png')

# ── ③ 頭頂放大：殘留多的一位 vs 乾淨的一位 ───────────────────────────────────
from check_top_residue import split_top, panel, GREEN, RED as RRED
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
EX = [('MRS0381-2', '殘留多的人'), ('T054', '乾淨的人')]      # 同 check_top_residue.py --example
npz = {r['subject']: r['npz'] for r in rows}
fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), facecolor=PAPER)
for ax, (sid, who) in zip(axes, EX):
    d = np.load(npz[sid])
    v, s = d['vol'], d['seg'].astype(int)
    rr, mm = split_top(v, s)
    panel(ax, v, s, mm, '%s（頭頂殘留 %.1f mm）' % (who, rr['total_mm']), size=(80, 36), overlay=True)
    ax.title.set_fontsize(15)
    ax.title.set_fontweight('bold')
fig.legend(handles=[Patch(facecolor=to_rgba(GREEN, 0.6), label='FreeSurfer 畫成皮質'),
                    Line2D([], [], color=RRED, lw=2.5, label='程式算成「殘留」')],
           loc='lower center', ncol=2, frameon=False, fontsize=13.5)
fig.tight_layout(rect=[0, 0.1, 1, 1])
save(fig, '1014_top_example.png')
