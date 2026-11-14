# -*- coding: utf-8 -*-
"""Part 3 下一步（第 28 頁）之補充圖（11-11 使用者：「解釋 28 頁時想跟老師說動哪，轉到 p31、p32，老師比較喜歡看圖」）
→ models\\deck_charts\\1111_detail_rcn.png、1111_detail_dwp.png、1111_detail_comb.png

四項原作細節（編號同第 28 頁）：
  ① 每級輸出 ×1/n（RCN 原作程式 flow_multiplier）        ② 越界取邊界值（RCN 論文 §3.1、程式 tf.clip_by_value）
  ③ 合成時 detach（DWP 原作 vxm_pwc.py 第 97 行）        ④ 粗層也算損失（DWP 論文 Implementation Details、run_iter.py）
位置依 ASD/arch.py、ASD/train_arch.py（11-11 重讀）：
  ① 合併版：乘在每一級「每一層」之速度場（VxmPyramid.fields 之 v = v * mult）→ 本研究自訂
  ② 合併版：只用於兩級之間（warp m、合成兩級形變、最後搬 m；VxmCascade.transformer）；每一級內部仍補 0
  ③ 合併版：每一級內每一層合成時（U = u + warp(U, u.detach())）
  ④ 合併版：只算第 2 級之 1/16、1/8、1/4 三層，權重 1/16、1/8、1/4（train_arch.py dwp_sublevel_loss）→ 本研究自訂
  平滑項：每一級每一層之速度場都罰（2 × 5 = 10 項），新舊合併版相同
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image

from style import J, INK, MUTED, PAPER, RULE, W, save

RCN_C, DWP_C, OWN_C = '#2F7FD0', '#0E7C7B', '#D08C2B'


def badge(ax, x, y, n, col, r=None, fs=13):
    ax.add_patch(Circle((x, y), r, fc=col, ec='white', lw=1.5, zorder=10))
    ax.text(x, y, n, ha='center', va='center', fontsize=fs, color='white', fontweight='bold', zorder=11)


def legend_strip(ax, x0, y, items, fs=11):
    for i, (n, col, t) in enumerate(items):
        xx = x0[i]
        ax.add_patch(Circle((xx, y), 0, fc='none'))
        ax.text(xx, y, n, ha='left', va='center', fontsize=fs + 1, color=col, fontweight='bold')
        ax.text(xx + 0.022, y, t, ha='left', va='center', fontsize=fs, color=INK)


# ════ RCN 原作圖：① ② ════
im = np.asarray(Image.open(J('models', 'deck_charts', '1028_pr_rcn_fig.png')).convert('RGB'))
H0, W0 = im.shape[:2]
EXT = 95                                           # 底下加一條圖例（像素）
fig = plt.figure(figsize=(W, W * (H0 + EXT) / W0))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(im, extent=(0, W0, H0, 0))
ax.set_xlim(0, W0); ax.set_ylim(H0 + EXT, 0); ax.axis('off')
R = 21
for x, y in ((382, 238), (886, 238)):              # Cascade f_k → Flow Field φ_k：網路輸出乘 1/n
    badge(ax, x, y, '①', RCN_C, R)
for x, y in ((489, 128), (992, 128), (1463, 126)):  # 搬移動影像之 warp（∘）
    badge(ax, x, y, '②', RCN_C, R)
ax.text(18, H0 + 48, '①', fontsize=14, color=RCN_C, fontweight='bold', va='center')
ax.text(58, H0 + 48, '每級網路之輸出乘 1/n（原作程式；論文未寫）', fontsize=12, color=INK, va='center')
ax.text(960, H0 + 48, '②', fontsize=14, color=RCN_C, fontweight='bold', va='center')
ax.text(1000, H0 + 48, '搬移動影像時，超出影像之座標取邊界值（論文 §3.1）', fontsize=12, color=INK, va='center')
save(fig, '1111_detail_rcn.png')

# ════ DWP 原作圖：③ ④（只標 Dual＋Warping＋Pyramid 用到的那一列）════
im = np.asarray(Image.open(J('models', 'deck_charts', '1028_pr_dwp_fig.png')).convert('RGB'))
H0, W0 = im.shape[:2]
EXT = 150
fig = plt.figure(figsize=(W * 0.52, W * 0.52 * (H0 + EXT) / W0))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(im, extent=(0, W0, H0, 0))
ax.set_xlim(0, W0); ax.set_ylim(H0 + EXT, 0); ax.axis('off')
ax.add_patch(FancyBboxPatch((592, 30), 868, 222, boxstyle='round,pad=0,rounding_size=12', fc='none', ec=DWP_C, lw=2.5, zorder=9))
R = 26
badge(ax, 1345, 160, '③', DWP_C, R, fs=15)          # Composition ⊙（在往下箭頭右邊）
badge(ax, 1398, 132, '④', DWP_C, R, fs=15)          # 每層輸出之總形變 φ^{l-1}（在方塊上方）
ax.text(20, H0 + 50, '③', fontsize=15, color=DWP_C, fontweight='bold', va='center')
ax.text(72, H0 + 50, '每層合成形變時，δ 只當取樣座標、不回傳梯度（detach；原作程式）', fontsize=11.5, color=INK, va='center')
ax.text(20, H0 + 112, '④', fontsize=15, color=DWP_C, fontweight='bold', va='center')
ax.text(72, H0 + 112, '粗層（1/16、1/8、1/4）之形變亦計相似度與平滑項，權重 1/16、1/8、1/4', fontsize=11.5, color=INK, va='center')
save(fig, '1111_detail_dwp.png')

# ════ 本研究之合併版：四項細節之位置 ════
HH = 5.2
fig = plt.figure(figsize=(W, HH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, HH); ax.axis('off')


def box(x, y, w, h, t, ec=INK, fc='white', fs=10.5, lw=1.4, ls='-', tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.07', fc=fc, ec=ec, lw=lw, ls=ls))
    if t:
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=tc, linespacing=1.3,
                fontweight='bold' if bold else 'normal')


def arrow(x1, y1, x2, y2, col=MUTED, lw=1.2, head=True):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>' if head else '-', mutation_scale=11, color=col, lw=lw))


def node(x, y, t, col=INK):
    ax.add_patch(Circle((x, y), 0.16, fc='white', ec=col, lw=1.6, zorder=5))
    ax.text(x, y, t, ha='center', va='center', fontsize=10, color=col, zorder=6)


def own_box(x, y, w, h):                            # 本研究自訂之規則（琥珀色虛線框）
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc='none', ec=OWN_C, lw=1.8, ls=(0, (4, 2)), zorder=7))


YL, YN, YF = 4.55, 4.1, 0.35                        # moving 線、warp 節點、fixed 線
YS, SH = 1.35, 2.45                                 # 級的底、高
LV = ['1/16', '1/8', '1/4', '1/2', '1']
box(0.1, YL - 0.25, 0.85, 0.5, r'Moving $m$')
box(0.1, YF - 0.25, 0.85, 0.5, r'Fixed $f$')


def stage(x0, title, sub_loss):
    w = 4.0
    box(x0, YS, w, SH, '', ec=DWP_C, fc='#F2F8F8', lw=1.8)
    ax.text(x0 + 0.12, YS + SH - 0.12, title, ha='left', va='top', fontsize=11, color=DWP_C, fontweight='bold')
    xs = [x0 + 0.3 + i * 0.8 for i in range(5)]
    yc = YS + 1.0
    for i, (x, l) in enumerate(zip(xs, LV)):
        box(x - 0.25, yc - 0.25, 0.5, 0.5, l, fs=9.5)
        if i < 4:
            node(x + 0.4, yc, '⊚')
            badge(ax, x + 0.4, yc - 0.38, '③', DWP_C, 0.12, fs=9)
    # ① 每一層之速度場 ×1/2：括號在五層上方
    ax.plot([xs[0] - 0.25, xs[0] - 0.25, xs[-1] + 0.25, xs[-1] + 0.25], [yc + 0.34, yc + 0.44, yc + 0.44, yc + 0.34], color=RCN_C, lw=1.6)
    badge(ax, xs[0] - 0.05, yc + 0.7, '①', RCN_C, 0.15, fs=11)
    ax.text(xs[0] + 0.16, yc + 0.7, '每一層之速度場 ×1/2', ha='left', va='center', fontsize=9.5, color=RCN_C)
    own_box(x0 + 0.03, yc + 0.34, w - 0.06, 0.6)
    ax.text(x0 + w / 2, YS + 0.17, '③ 每一層合成形變時 detach', ha='center', va='center', fontsize=9.5, color=DWP_C)
    if sub_loss:                                    # ④ 粗層損失：只在第 2 級之 1/16、1/8、1/4
        yb = YS - 0.1
        ax.plot([xs[0] - 0.25, xs[0] - 0.25, xs[2] + 0.25, xs[2] + 0.25], [yb + 0.07, yb - 0.04, yb - 0.04, yb + 0.07], color=DWP_C, lw=1.6)
        badge(ax, xs[0] - 0.05, yb - 0.3, '④', DWP_C, 0.15, fs=11)
        ax.text(xs[0] + 0.16, yb - 0.3, '粗層亦計損失（權重 1/16、1/8、1/4）', ha='left', va='center', fontsize=9.5, color=DWP_C)
        own_box(x0 + 0.03, yb - 0.55, 2.95, 0.6)
    return x0 + w, YS + SH, yc


S1x, S2x = 1.35, 6.45
e1, top, yc = stage(S1x, '第 1 級：雙流金字塔', False)
e2, _, _ = stage(S2x, '第 2 級：雙流金字塔（僅此級加 ④）', True)
XW, XC = 5.9, 10.85
# moving 線：m 一路往右，分別往下接到第 1 級、第一個 warp、最後的 warp
ax.plot([0.95, XC], [YL, YL], color=MUTED, lw=1.2)
arrow(1.6, YL, 1.6, top + 0.02)
arrow(XW, YL, XW, YN + 0.17)
arrow(XC, YL, XC, YN + 0.17)
# fixed 線：往上接到兩級
ax.plot([0.95, S2x + 3.9], [YF, YF], color=MUTED, lw=1.2)
arrow(S1x + 3.9, YF, S1x + 3.9, YS - 0.02)
arrow(S2x + 3.9, YF, S2x + 3.9, YS - 0.02)
# 兩級之間：Φ1 → warp（②）→ m∘Φ1 → 第 2 級
arrow(e1, yc, XW, yc, head=False)
ax.text((e1 + XW) / 2, yc + 0.06, r'$\Phi_1$', ha='center', va='bottom', fontsize=10, color=INK)
arrow(XW, yc, XW, YN - 0.17)
node(XW, YN, 'w')
badge(ax, XW - 0.32, YN, '②', RCN_C, 0.15, fs=11)
arrow(XW + 0.13, YN - 0.1, S2x + 0.35, top + 0.02)
ax.text(XW + 0.32, YN + 0.08, r'$m\circ\Phi_1$', ha='left', va='bottom', fontsize=10, color=INK)
# 第 2 級之後：合成 Φ = Φ1∘φ2（②）→ 搬 m（②）→ 相似度
arrow(e2, yc, XC - 0.17, yc)
ax.text(e2 + 0.08, yc + 0.06, r'$\varphi_2$', ha='left', va='bottom', fontsize=10, color=INK)
node(XC, yc, '⊚')
badge(ax, XC + 0.33, yc, '②', RCN_C, 0.15, fs=11)
ax.text(XC, yc - 0.3, r'$\Phi=\Phi_1\circ\varphi_2$', ha='center', va='top', fontsize=10, color=INK)
arrow(XC, yc + 0.17, XC, YN - 0.17)
node(XC, YN, 'w')
badge(ax, XC - 0.32, YN, '②', RCN_C, 0.15, fs=11)
arrow(XC + 0.17, YN, 11.18, YN)
box(11.2, YN - 0.45, 0.88, 0.9, '相似度\n（僅最終\n輸出）', fs=9)
# 圖例
ax.text(0.1, 5.0, '●', fontsize=12, color=RCN_C, va='center'); ax.text(0.33, 5.0, 'RCN 細節（--rcn）：① ②', fontsize=10.5, color=INK, va='center')
ax.text(3.0, 5.0, '●', fontsize=12, color=DWP_C, va='center'); ax.text(3.23, 5.0, 'DWP 細節（--dwp）：③ ④', fontsize=10.5, color=INK, va='center')
ax.plot([5.95, 6.4], [5.0, 5.0], color=OWN_C, lw=1.8, ls=(0, (4, 2)))
ax.text(6.5, 5.0, '合併時本研究自訂之規則（原作無此組合）', fontsize=10.5, color=INK, va='center')
ax.text(W - 0.1, 0.1, '平滑項：每一級之每一層皆罰（2 × 5 = 10 項），新舊合併版相同', ha='right', va='center', fontsize=9.5, color=MUTED)
save(fig, '1111_detail_comb.png')
