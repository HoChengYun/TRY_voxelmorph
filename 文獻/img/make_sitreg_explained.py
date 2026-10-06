# -*- coding: utf-8 -*-
"""教學圖：SITReg 怎麼做到（幾乎）不擠爆、正反向一致。

左上：沒有上限時，2 號點一步走太遠、越過 3 號點，順序翻過來＝擠爆。
左下：SITReg 每一步都有上限（相鄰兩點的位移差 < 兩點的距離），所以越不過鄰居；很多小步接起來一樣能走很遠。
右：兩張影像各走一半、在中間會合，A→B 跟 B→A 一定互為反函數（這招是為了對稱，不是為了不擠爆）。

點的位置是舉例。論文：Honkamaa & Marttinen, SITReg: Multi-resolution architecture for symmetric,
inverse consistent, and topology preserving image registration, MELBA 2024（arXiv 2303.10211）。

輸出到同資料夾的 sitreg_explained.png。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
GRAY, CORAL, RED, TEAL, PURPLE = '#888780', '#D85A30', '#D62728', '#0E7C7B', '#7F77DD'

X0, SP = 14.0, 11.0           # 1 號點的位置、點的間距


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.6, head=10):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=head,
                                shrinkA=0, shrinkB=0))


def rows_of_points(ax, rows, ys, labels):
    """rows：每一列 4 個點的位置（以間距為單位）；2 號點畫成橘色。"""
    for k in range(len(rows) - 1):
        for p in range(4):
            ax.plot([X0 + rows[k][p] * SP, X0 + rows[k + 1][p] * SP], [ys[k] - 1.1, ys[k + 1] + 1.1],
                    color=CORAL if p == 1 else GRAY, lw=1.8 if p == 1 else 1.0)
    for r, y, lab in zip(rows, ys, labels):
        for p in range(4):
            ax.plot([X0 + r[p] * SP], [y], 'o', ms=11, color=CORAL if p == 1 else '#B4B2A9',
                    markeredgecolor='white', markeredgewidth=0.8)
        ax.text(X0 + 3.9 * SP, y, lab, va='center', fontsize=11, color=MUTED)


fig = plt.figure(figsize=(12.5, 8.4))
fig.patch.set_facecolor(PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 125)
ax.set_ylim(0, 84)
ax.set_aspect('equal')
ax.axis('off')

ax.text(3, 80.5, 'SITReg 怎麼做到不擠爆：每一步都有上限，再接很多步', fontsize=17, fontweight='bold',
        color=INK, va='center')
ax.text(3, 76.6, '擠爆＝相鄰的點走完之後順序翻過來（Jacobian 小於等於 0）。用一排點示意（位置是舉例）',
        fontsize=11, color=MUTED, va='center')

# 左上：沒有上限
ax.text(3, 70, '沒有上限', fontsize=13, fontweight='bold', color=INK, va='center')
ax.text(3, 66.8, '一步走太大', fontsize=10.5, color=MUTED, va='center')
for p in range(4):
    ax.text(X0 + p * SP, 72.6, '%d' % (p + 1), ha='center', va='center', fontsize=10.5, color=MUTED)
rows_of_points(ax, [[0, 1, 2, 3], [0, 2.55, 2, 3]], [69.5, 61.5], ['起點', '走一步'])
ax.text(X0 + 1.5 * SP, 56.5, '2 號越過 3 號，順序翻過來 ＝ 擠爆', ha='center', va='center', fontsize=11.5,
        fontweight='bold', color=RED)

# 左下：SITReg 每步有上限
ax.plot([2, 72], [52.5, 52.5], color='#DDDCD6', lw=1)
ax.text(3, 47.5, 'SITReg', fontsize=13, fontweight='bold', color=TEAL, va='center')
ax.text(3, 44.3, '每步有上限', fontsize=10.5, color=MUTED, va='center')
ax.text(3, 41.6, '走好幾步', fontsize=10.5, color=MUTED, va='center')
for p in range(4):
    ax.text(X0 + p * SP, 50.1, '%d' % (p + 1), ha='center', va='center', fontsize=10.5, color=MUTED)
ROWS = [[0, 1, 2, 3], [0, 1.44, 2.25, 3.09], [0, 1.88, 2.5, 3.16], [0, 2.31, 2.75, 3.25]]
rows_of_points(ax, ROWS, [47, 39, 31, 23], ['起點', '第 1 步', '第 2 步', '第 3 步'])
ax.text(X0 + 1.5 * SP, 17.6, '每一步：相鄰兩點的位移差 < 兩點的距離 → 越不過鄰居', ha='center', va='center',
        fontsize=11.5, fontweight='bold', color=TEAL)
ax.text(X0 + 1.5 * SP, 14.2, '很多小步接起來，一樣能走很遠（2 號從 2 走到 3.3 附近，順序都沒翻）',
        ha='center', va='center', fontsize=10.5, color=INK)
ax.text(3, 8.8, '上限怎麼做到：網路在比較粗的「控制點」上輸出位移，先經過 tanh 壓在上限以內', fontsize=10.5,
        color=MUTED, va='center')
ax.text(3, 5.8, '（理論上限的 0.99 倍），再用三次 B-spline 平滑插值到全尺寸 → 數學上保證這一步可以倒回去',
        fontsize=10.5, color=MUTED, va='center')
ax.text(3, 2.8, '由粗到細 5 層，每層一小步；「不會翻的變換」接在一起也不會翻', fontsize=10.5, color=MUTED,
        va='center')

# 右：兩張影像各走一半
ax.add_patch(FancyBboxPatch((76, 3), 47, 70.5, boxstyle='round,pad=0.4,rounding_size=1.5',
                            facecolor='#F1F0EA', edgecolor='none'))
ax.text(78.5, 70, '另一招：兩張各走一半，在中間會合', fontsize=13, fontweight='bold', color=INK, va='center')
ax.text(78.5, 66.6, '（這招是為了「對稱」，不是為了不擠爆）', fontsize=10.5, color=MUTED, va='center')


def box(x, y, text, color):
    ax.add_patch(FancyBboxPatch((x - 6.5, y - 3.2), 13, 6.4, boxstyle='round,pad=0.2,rounding_size=1',
                                facecolor=color, alpha=0.25, edgecolor=color, lw=1.2))
    ax.text(x, y, text, ha='center', va='center', fontsize=12, fontweight='bold', color=INK)


box(86, 57, '影像 A', PURPLE)
box(114, 57, '影像 B', CORAL)
box(100, 41, '中間', TEAL)
arrow(ax, 89, 53.6, 96.5, 44.6, color=PURPLE, lw=1.8, head=12)
arrow(ax, 111, 53.6, 103.5, 44.6, color=CORAL, lw=1.8, head=12)
ax.text(89.5, 48, '走一半', ha='right', va='center', fontsize=10.5, color=MUTED)
ax.text(110.5, 48, '走一半', ha='left', va='center', fontsize=10.5, color=MUTED)
ax.text(99.5, 31.5, r'$f_{A\to B} \;=\; d_{A\to \mathrm{mid}} \circ \left(d_{B\to \mathrm{mid}}\right)^{-1}$',
        ha='center', va='center', fontsize=16, color=INK)
ax.text(78.5, 25.5, 'A→B ＝「A 那一半」接上「B 那一半倒過來」', fontsize=10.5, color=INK, va='center')
ax.text(78.5, 22.2, '把 A、B 對調，算出來剛好是反方向', fontsize=10.5, color=INK, va='center')
ax.text(78.5, 18.9, '→ A→B 跟 B→A 一定互為反函數', fontsize=10.5, color=INK, va='center')
ax.text(78.5, 13.6, '「倒過來」要算反函數：論文另外做一層專門算', fontsize=10.5, color=MUTED, va='center')
ax.text(78.5, 10.3, '（反覆逼近求解；不存中間過程，省記憶體）', fontsize=10.5, color=MUTED, va='center')
ax.text(78.5, 5.8, '代價：一次對位要算 8 次反函數，0.37 秒', fontsize=10.5, color=MUTED, va='center')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sitreg_explained.png')
fig.savefig(out, dpi=115, facecolor=PAPER)
print('寫出', out)
