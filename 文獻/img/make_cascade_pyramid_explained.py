# -*- coding: utf-8 -*-
"""文獻/cascade與pyramid原理筆記.md 用的兩張教學圖（2026-10-21）。

  cp_compose.png   為什麼兩段形變要「合成」、不能直接相加（一維、照著數字走一遍）
  cp_reach.png     同一個 12 mm 的位移，在每一層的格子上是幾格（粗的層看得到大位移的原因）

另外 cp_arch_unet_vs_pyramid.png 是從 models\\deck_charts 複製的（10/14 簡報 make_arch.py 畫的結構圖，跟現在的程式一致）；
cascade 與 decoder 單層的新版圖（紅字標出 10-21 照原作補上的部分）在 make_cp_arch_v2.py。

用法：python 文獻\\img\\make_cascade_pyramid_explained.py
"""
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
TEAL, BLUE, RED, AMBER = '#0E7C7B', '#2F7FD0', '#B03A2E', '#D08C2B'


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.savefig(p, dpi=130, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


# ── 1. 合成 vs 相加 ─────────────────────────────────────────────────
# 第一段形變 U1：位置不同、位移不同（U1(10) = 3、U1(12) = 5）；第二段在 x = 10 說「往右 2 格」。
x0, u2 = 10, 2
U1 = lambda x: 3 + (x - 10)          # U1(10) = 3, U1(12) = 5
fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=PAPER)
ax.set_facecolor(PAPER)
rows = {'f': 2.0, 'mid': 1.0, 'm': 0.0}
for y, lab in [(2.0, '固定影像 f 的格子（我們要知道每一格對應到 m 的哪裡）'),
               (1.0, '第一段 φ₁ 拉過來的影像 m∘φ₁ 的格子（第二段網路看到的是它）'),
               (0.0, '移動影像 m 的格子')]:
    ax.plot([4, 22], [y, y], color=MUTED, lw=1)
    for t in range(4, 23):
        ax.plot([t, t], [y - 0.05, y + 0.05], color=MUTED, lw=0.8)
    ax.text(4, y + 0.18, lab, fontsize=9.5, color=MUTED, va='bottom')


def arrow(xa, ya, xb, yb, c, lab, dy=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle='-|>', mutation_scale=14, color=c, lw=2, ls=ls))
    ax.text((xa + xb) / 2 + 0.15, (ya + yb) / 2 + dy, lab, color=c, fontsize=10.5, va='center')


ax.plot(x0, 2.0, 'o', color=INK, ms=8)
ax.text(x0, 2.42, 'x = %d' % x0, ha='center', fontsize=10.5, color=INK)
# 第二段：在 x 往右 u2
arrow(x0, 1.92, x0 + u2, 1.08, BLUE, r'① 第二段 $u_2(x)=+%d$' % u2, dy=0.05)
ax.plot(x0 + u2, 1.0, 'o', color=BLUE, ms=7)
# 第一段：在 x + u2 = 12 取 U1
arrow(x0 + u2, 0.92, x0 + u2 + U1(x0 + u2), 0.08, TEAL,
      r'② 在 $x+u_2=%d$ 取第一段：$U_1(%d)=+%d$' % (x0 + u2, x0 + u2, U1(x0 + u2)), dy=0.05)
ax.plot(x0 + u2 + U1(x0 + u2), 0.0, 'o', color=TEAL, ms=8)
ax.text(x0 + u2 + U1(x0 + u2) + 0.3, -0.42, '合成：%d + %d + %d = %d ✓' % (x0, u2, U1(x0 + u2), x0 + u2 + U1(x0 + u2)),
        ha='left', fontsize=10.5, color=TEAL, fontweight='bold')
# 直接相加：U1 取在 x = 10
ax.plot(x0 + u2 + U1(x0), 0.0, 'x', color=RED, ms=10, mew=2.5)
ax.text(x0 + u2 + U1(x0) - 0.3, -0.42, '直接相加：%d + %d + %d = %d ✗' % (x0, u2, U1(x0), x0 + u2 + U1(x0)),
        ha='right', fontsize=10.5, color=RED)
ax.text(22.3, 1.0, r'$U(x)=u_2(x)+U_1(x+u_2(x))$' + '\n' + r'第一段要在「第二段指到的地方」取，不是在 $x$ 取',
        fontsize=11, color=INK, va='center')
ax.set_xlim(3.5, 31.5)
ax.set_ylim(-0.75, 2.75)
ax.axis('off')
ax.set_title('為什麼兩段形變要「合成」：第一段的位移跟位置有關（這裡 $U_1(10)=3$、$U_1(12)=5$）', fontsize=11.5, loc='left', color=INK)
save(fig, 'cp_compose.png')

# ── 2. 同一個 12 mm，在每一層是幾格 ────────────────────────────────────
mm = 12.0
levels = [('原尺寸', 1), ('1/2', 2), ('1/4', 4), ('1/8', 8), ('1/16', 16)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7), facecolor=PAPER, gridspec_kw={'width_ratios': [1.25, 1]})
for a in (a1, a2):
    a.set_facecolor(PAPER)
vals = [mm / s for _, s in levels]
cols = [RED if v > 1.5 else (AMBER if v > 1 else TEAL) for v in vals]
a1.bar(range(len(levels)), vals, color=cols)
for i, v in enumerate(vals):
    a1.text(i, v + 0.2, '%g 格' % v, ha='center', fontsize=10.5, color=INK)
a1.axhline(1, color=MUTED, ls='--', lw=1.2)
a1.set_xticks(range(len(levels)))
a1.set_xticklabels(['%s\n（1 格 = %d mm）' % (n, s) for n, s in levels], fontsize=9.5)
a1.set_ylabel('12 mm 的位移 = 幾格', fontsize=10)
a1.set_title('(a) 同一個 12 mm 的位移，在越粗的格子上越「小」\n'
             '虛線：一次 3×3×3 卷積看得到的範圍（±1 格）', fontsize=11, loc='left', color=INK)
for sp in ('top', 'right'):
    a1.spines[sp].set_visible(False)
# (b) 由粗到細：每一層只修剩下的
left = [12.0]
fixes = [('1/16', 8.0), ('1/8', 2.5), ('1/4', 1.0), ('1/2', 0.4), ('原尺寸', 0.1)]
rem = 12.0
xs, ys, labs = [], [], []
for i, (n, fx) in enumerate(fixes):
    a2.barh(i, fx, left=12.0 - rem, color=[TEAL, BLUE, AMBER, RED, '#7A4FA0'][i], height=0.6)
    a2.text(12.0 - rem + fx + 0.15, i, '%s 層修 %g mm' % (n, fx), va='center', fontsize=9.5, color=INK)
    rem -= fx
a2.axvline(12.0, color=INK, ls=':', lw=1)
a2.text(12.0, 4.75, '真正的位移 12 mm', ha='right', fontsize=9.5, color=INK)
a2.set_xlim(0, 15.5)
a2.set_ylim(4.9, -0.6)
a2.set_yticks([])
a2.set_xlabel('累積的位移（mm）', fontsize=10)
a2.set_title('(b) 由粗到細：粗的層先移大部分，細的層只修剩下的（示意）', fontsize=11, loc='left', color=INK)
for sp in ('top', 'right', 'left'):
    a2.spines[sp].set_visible(False)
fig.tight_layout()
save(fig, 'cp_reach.png')

# ── 3. 從 deck_charts 複製三張架構圖 ───────────────────────────────────
for src, dst in [('1014_arch_unet_vs_pyramid.png', 'cp_arch_unet_vs_pyramid.png')]:
    shutil.copyfile(os.path.join(ROOT, 'models', 'deck_charts', src), os.path.join(HERE, dst))
    print('->', os.path.join(HERE, dst), '（複製自 models\\deck_charts\\%s）' % src)
