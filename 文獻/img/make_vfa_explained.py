# -*- coding: utf-8 -*-
"""教學圖：VFA（Vector Field Attention）怎麼找對應點。

上：在一個點上做的事（2D 的 3×3 示意，論文是 3D 的 3×3×3＝27 格）
    固定影像這一點的特徵，跟移動影像同一位置附近每一格的特徵比「有多像」→ softmax 變成權重，
    每一格事先存好一個固定的方向（＝對應點在那一格時的位移），權重 × 方向加起來＝這一點的位移。
    這一步沒有要學的參數，要學的只有前面抽特徵的網路。
下：一層只看得到 ±1 格，所以由粗到細做 5 層；論文的算法：每往上一層「×2 再 +2」，3 → 8 → 18 → 38 → 78。

權重是舉例用的數字。論文：Liu et al., Vector field attention for deformable image registration,
J. Med. Imaging 11(6) 064001, 2024（arXiv 2407.10209）。

輸出到同資料夾的 vfa_explained.png。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
PURPLE, AMBER, AMBER_EDGE, GRAY, CORAL, TEAL = '#7F77DD', '#EF9F27', '#BA7517', '#888780', '#D85A30', '#0E7C7B'

C = 8.0                       # 一格的邊長
W = [[0.02, 0.05, 0.55],      # 權重（舉例），上到下、左到右；加起來 = 1
     [0.03, 0.10, 0.20],
     [0.01, 0.02, 0.02]]
YC = 53.0                     # 上半部格子的中心高度
T1, T2 = YC + 17.4, YC + 14.2  # 每一欄的標題、副標
B1, B2 = YC - 15.0, YC - 18.2  # 每一欄下面的兩行


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.6, head=10):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=head,
                                shrinkA=0, shrinkB=0))


def grid(ax, cx, cy, face=None, alpha=None, edge=GRAY, lw=0.8):
    for i in range(3):
        for j in range(3):
            x = cx + (j - 1.5) * C
            y = cy + (0.5 - i) * C
            fc = face if face is not None else 'none'
            a = alpha[i][j] if alpha is not None else 1.0
            ax.add_patch(Rectangle((x, y), C, C, facecolor=fc, alpha=a, edgecolor='none'))
            ax.add_patch(Rectangle((x, y), C, C, facecolor='none', edgecolor=edge, lw=lw))


def heads(ax, x, title, sub, low1, low2):
    ax.text(x, T1, title, ha='center', va='center', fontsize=12.5, fontweight='bold', color=INK)
    ax.text(x, T2, sub, ha='center', va='center', fontsize=10.5, color=MUTED)
    if low1:
        ax.text(x, B1, low1, ha='center', va='center', fontsize=11, color=INK)
    if low2:
        ax.text(x, B2, low2, ha='center', va='center', fontsize=10.5, color=MUTED)


fig = plt.figure(figsize=(12.5, 8.4))
fig.patch.set_facecolor(PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 125)
ax.set_ylim(0, 84)
ax.set_aspect('equal')
ax.axis('off')

ax.text(3, 81, 'VFA 怎麼找對應點：拿去「比對」，不是讓網路直接猜位移', fontsize=17, fontweight='bold',
        color=INK, va='center')
ax.text(3, 77.4, '在一個點上做的事（用 2D 的 3×3 示意，論文是 3D 的 3×3×3＝27 格；權重是舉例的數字）',
        fontsize=11, color=MUTED, va='center')

# ① 固定影像的這一點
heads(ax, 11, '固定影像', '（不動的那張）', None, None)
ax.add_patch(Rectangle((11 - C / 2, YC - C / 2), C, C, facecolor=PURPLE, edgecolor='none'))
ax.text(11, YC - 7.6, '這一點的特徵', ha='center', va='center', fontsize=11, color=INK)
ax.text(11, YC - 10.8, r'$F(x)$', ha='center', va='center', fontsize=13, color=INK)
arrow(ax, 17, YC, 26, YC)

# ② 移動影像附近 3×3，比有多像 → 權重
cx2 = 40
wmax = max(max(r) for r in W)
grid(ax, cx2, YC, face=AMBER, alpha=[[0.08 + 0.85 * w / wmax for w in r] for r in W], edge=AMBER_EDGE)
for i in range(3):
    for j in range(3):
        w = W[i][j]
        ax.text(cx2 + (j - 1) * C, YC + (1 - i) * C, '%.2f' % w, ha='center', va='center',
                fontsize=12 if w == wmax else 10.5, fontweight='bold' if w == wmax else 'normal', color=INK)
heads(ax, cx2, '移動影像：同一位置附近 3×3', '每一格的特徵跟 F(x) 比有多像',
      '→ softmax → 權重（加起來＝1）', '越像，權重越大')

ax.text(58.5, YC, '×', ha='center', va='center', fontsize=24, color=INK)

# ③ 每一格事先存好的方向（固定，不用學）
cx3 = 77
grid(ax, cx3, YC)
for i in range(3):
    for j in range(3):
        dx, dy = (j - 1), (1 - i)
        if dx == 0 and dy == 0:
            continue
        arrow(ax, cx3, YC, cx3 + dx * C * 0.82, YC + dy * C * 0.82, color=GRAY, lw=1.3, head=9)
ax.plot([cx3], [YC], 'o', color=GRAY, ms=4)
heads(ax, cx3, '每一格事先存好的方向', '＝「對應點在這一格」時的位移',
      '固定的，不用學', '（論文叫「放射狀向量場」）')

ax.text(95.5, YC, '＝', ha='center', va='center', fontsize=24, color=INK)

# ④ 加權平均 ＝ 位移
cx4 = 112
grid(ax, cx4, YC, edge='#CFCFCB', lw=0.6)
ax.add_patch(Rectangle((cx4 - C / 2, YC - C / 2), C, C, facecolor=PURPLE, alpha=0.35, edgecolor='none'))
ux = sum(W[i][j] * (j - 1) for i in range(3) for j in range(3))
uy = sum(W[i][j] * (1 - i) for i in range(3) for j in range(3))
arrow(ax, cx4, YC, cx4 + ux * C, YC + uy * C, color=CORAL, lw=3, head=16)
ax.plot([cx4], [YC], 'o', color=CORAL, ms=5)
heads(ax, cx4, '加權平均', '＝這一點的位移', '大部分指向最像的那格', '（右上，權重 0.55）')

# 公式
ax.text(62.5, 28.6, r'$u(x) \;=\; \sum_{k=1}^{27}\; w_k(x)\; r_k, \qquad '
        r'w_k(x) \;=\; \mathrm{softmax}_k\left( F(x)\cdot M_k(x) \right)$',
        ha='center', va='center', fontsize=17, color=INK)
ax.text(62.5, 22.4, '其中　F(x)：固定影像在這一點的特徵　　M_k(x)：移動影像第 k 個鄰居的特徵　　'
        'r_k：第 k 格存好的方向（固定）',
        ha='center', va='center', fontsize=10.5, color=MUTED)

# 下半部：由粗到細 5 層
ax.add_patch(FancyBboxPatch((2, 1.2), 121, 18.6, boxstyle='round,pad=0.4,rounding_size=1.5',
                            facecolor='#F1F0EA', edgecolor='none'))
ax.text(4, 17.6, '一層只看得到 ±1 格，所以由粗到細做 5 層', fontsize=13, fontweight='bold', color=INK,
        va='center')
ax.text(121, 17.6, '方框裡的數字＝每邊看得到幾格（論文的算法）', fontsize=10, color=MUTED, va='center',
        ha='right')
ax.text(4, 14.5, '每一層：先用上一層的結果把移動影像的特徵拉過來 → 卷積修正一下 → 再比一次 ±1 格 → '
        '跟上一層的形變接起來', fontsize=10.5, color=MUTED, va='center')
LEVELS = [('第 5 層', '1/16 大小', 3), ('第 4 層', '1/8', 8), ('第 3 層', '1/4', 18),
          ('第 2 層', '1/2', 38), ('第 1 層', '原尺寸', 78)]
xs = [14, 38, 62, 86, 110]
for (name, size, reach), x in zip(LEVELS, xs):
    last = reach == 78
    ax.add_patch(FancyBboxPatch((x - 8.5, 2.4), 17, 9, boxstyle='round,pad=0.2,rounding_size=1',
                                facecolor=PAPER, edgecolor=TEAL if last else GRAY, lw=1.6 if last else 0.8))
    ax.text(x, 9.4, '%s（%s）' % (name, size), ha='center', va='center', fontsize=10.5, color=INK)
    ax.text(x, 5.3, ('%d×%d×%d' % (reach, reach, reach)) if last else ('%d 格' % reach), ha='center',
            va='center', fontsize=16 if last else 17, fontweight='bold', color=TEAL if last else INK)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, a + 9.2, 6.4, b - 9.2, 6.4, color=MUTED, lw=1.2, head=9)
    ax.text((a + b) / 2, 8.5, '×2 ＋2', ha='center', va='center', fontsize=9.5, color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vfa_explained.png')
fig.savefig(out, dpi=115, facecolor=PAPER)
print('寫出', out)
