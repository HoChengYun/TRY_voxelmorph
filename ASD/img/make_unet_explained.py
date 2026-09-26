# -*- coding: utf-8 -*-
"""手冊 §23 的教學圖：VoxelMorph 在做什麼、U-Net 長什麼樣、mix_wide 動了哪裡。

層的排法照 voxelmorph/torch/networks.py 的 Unet
（encoder 4 層 stride-2、decoder 4 層上採樣 + 跳接、再 3 個全尺寸 extras、最後 3 通道形變場）。
純手畫的示意圖，不讀任何資料。輸出到同資料夾的 unet_explained.png。
"""
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
INK, MUTED, PAPER, RULE = '#141A1D', '#5F6A6B', '#FAFAF8', '#C9C9C1'
TEAL, RUST, BOXF, GRAYF = '#0E7C7B', '#A34F1B', '#FFFFFF', '#ECECE7'

fig = plt.figure(figsize=(17.5, 10.2), facecolor=PAPER)


def box(ax, x, y, w, h, fc=BOXF, ec=INK, lw=1.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle='round,pad=0.006,rounding_size=0.012', fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, y0, x1, y1, color=INK, ls='-', lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=15,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


# ── ① 上：VoxelMorph 整條流程 ─────────────────────────────────────
ax = fig.add_axes([0.01, 0.80, 0.98, 0.17])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.text(0.0, 1.0, '① VoxelMorph 在做什麼', fontsize=16, fontweight='bold', va='top')
steps = [(0.06, '受試者的腦 M\n＋ 模板 A', GRAYF), (0.22, '疊在一起\n（2 張）', GRAYF),
         (0.38, 'U-Net\n（下面放大）', '#DCEFEE'), (0.55, '形變場 φ\n每個點往哪移', GRAYF),
         (0.72, '照 φ 把 M 捏過去\n→ 捏好的腦', GRAYF), (0.90, '打分數\n像不像 A ＋ 捏得平不平滑', GRAYF)]
for i, (x, t, fc) in enumerate(steps):
    box(ax, x, 0.40, 0.125 if i < 5 else 0.17, 0.52, fc=fc,
        ec=TEAL if i == 2 else INK, lw=2.4 if i == 2 else 1.3)
    ax.text(x, 0.40, t, ha='center', va='center', fontsize=12, fontweight='bold' if i == 2 else 'normal')
    if i < 5:
        nx = steps[i + 1][0]
        arrow(ax, x + 0.066, 0.40, nx - (0.066 if i < 4 else 0.088), 0.40)
ax.text(0.90, 0.02, '分數越好 → 回頭調 U-Net 的參數', ha='center', fontsize=10.5, color=MUTED)

# ── ② 左下：U-Net ─────────────────────────────────────────────────
ax = fig.add_axes([0.01, 0.02, 0.64, 0.74])
ax.set_xlim(-0.2, 1.0)
ax.set_ylim(-0.04, 1.02)
ax.axis('off')
ax.text(-0.2, 1.02, '② U-Net 長這樣　框裡的數字 = 通道數（這一層同時產生幾張特徵圖）',
        fontsize=15, fontweight='bold', va='top')
ax.text(-0.2, 0.965, '預設', color=TEAL, fontsize=13, fontweight='bold', va='top')
ax.text(-0.14, 0.965, '→', color=MUTED, fontsize=13, va='top')
ax.text(-0.10, 0.965, 'mix_wide 加寬後', color=RUST, fontsize=13, fontweight='bold', va='top')

ROW = {192: 0.84, 96: 0.64, 48: 0.46, 24: 0.28, 12: 0.10}
SIZE = {192: '192×224×192\n全尺寸', 96: '96×112×96', 48: '48×56×48', 24: '24×28×24', 12: '12×14×12'}
for k, y in ROW.items():
    ax.text(-0.19, y, SIZE[k], ha='left', va='center', fontsize=10.5, color=MUTED)
    if k != 12:
        ax.plot([-0.19, 0.98], [y - 0.09, y - 0.09], color=RULE, lw=0.6, ls=':')

W, H = 0.12, 0.105


def layer(x, y, name, a, b, changed=True, fc=BOXF):
    box(ax, x, y, W, H, fc=fc, ec=INK)
    ax.text(x, y + 0.022, name, ha='center', va='center', fontsize=10.5, color=MUTED)
    if changed:
        ax.text(x - 0.012, y - 0.022, str(a), ha='right', va='center', fontsize=13, fontweight='bold', color=TEAL)
        ax.text(x, y - 0.022, '→', ha='center', va='center', fontsize=11, color=MUTED)
        ax.text(x + 0.012, y - 0.022, str(b), ha='left', va='center', fontsize=13, fontweight='bold', color=RUST)
    else:
        ax.text(x, y - 0.022, '%s（不變）' % a, ha='center', va='center', fontsize=12, fontweight='bold', color=INK)


XE, XD = 0.07, 0.47
layer(XE, ROW[192], '輸入', 2, 2, changed=False, fc=GRAYF)
layer(XE, ROW[96], '縮小 1', 16, 32)
layer(XE, ROW[48], '縮小 2', 32, 64)
layer(XE, ROW[24], '縮小 3', 32, 64)
layer(XE, ROW[12], '縮小 4', 32, 64)
layer(XD, ROW[12], '放大 1', 32, 64)
layer(XD, ROW[24], '放大 2', 32, 64)
layer(XD, ROW[48], '放大 3', 32, 64)
layer(XD, ROW[96], '放大 4', 32, 64)
layer(XD, ROW[192], '收尾 1', 32, 64)
layer(0.62, ROW[192], '收尾 2', 16, 32)
layer(0.77, ROW[192], '收尾 3', 16, 32)
layer(0.92, ROW[192], '形變場 φ', 3, 3, changed=False, fc='#DCEFEE')

for a, b in ((192, 96), (96, 48), (48, 24), (24, 12)):
    arrow(ax, XE, ROW[a] - H / 2, XE, ROW[b] + H / 2)
arrow(ax, XE + W / 2, ROW[12], XD - W / 2, ROW[12])
for a, b in ((12, 24), (24, 48), (48, 96), (96, 192)):
    arrow(ax, XD, ROW[a] + H / 2, XD, ROW[b] - H / 2)
for x0, x1 in ((XD, 0.62), (0.62, 0.77), (0.77, 0.92)):
    arrow(ax, x0 + W / 2, ROW[192], x1 - W / 2, ROW[192])
for k in (192, 96, 48, 24):
    arrow(ax, XE + W / 2, ROW[k], XD - W / 2, ROW[k], color='#8C9A9B', ls='--', lw=1.3)

LX, LY = 0.63, 0.56
ax.text(LX, LY + 0.06, '怎麼看這張圖', fontsize=12.5, fontweight='bold', va='center')
arrow(ax, LX, LY, LX, LY - 0.07)
ax.text(LX + 0.04, LY - 0.035, '左邊往下：每次縮小一半\n　看的範圍越來越大（抓整體輪廓）', fontsize=10.5, va='center')
arrow(ax, LX, LY - 0.20, LX, LY - 0.13)
ax.text(LX + 0.04, LY - 0.165, '右邊往上：每次放大一倍\n　一路回到原尺寸（補回細節）', fontsize=10.5, va='center')
arrow(ax, LX - 0.02, LY - 0.27, LX + 0.03, LY - 0.27, color='#8C9A9B', ls='--', lw=1.3)
ax.text(LX + 0.05, LY - 0.27, '虛線（跳接）：左邊同尺寸的結果\n　直接抄到右邊，避免細節縮不見', fontsize=10.5, va='center')
ax.text(0.62, ROW[192] - 0.115, '↑ 這一排是全尺寸（826 萬格），最吃顯存：\n　一張 32 通道的特徵圖就約 1 GB',
        fontsize=10.5, color=RUST, va='top')

# ── ③ 右下：這次動了什麼 ───────────────────────────────────────────
ax = fig.add_axes([0.665, 0.02, 0.325, 0.74])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.text(0, 1.0, '③ mix_wide 動了什麼', fontsize=16, fontweight='bold', va='top')
ax.add_patch(FancyBboxPatch((0, 0.70), 1, 0.225, boxstyle='round,pad=0.01', fc='#F6E7DC', ec=RUST, lw=1.4))
ax.text(0.03, 0.905, '只動一件事', fontsize=13.5, fontweight='bold', color=RUST, va='top')
ax.text(0.03, 0.845, '每一層的通道數 × 2\n（輸入 2 張、輸出 3 張不變）', fontsize=12.5, va='top')
ax.text(0.03, 0.745, '參數 301,411 → 1,197,251（4 倍）', fontsize=11.5, color=MUTED, va='top')

ax.text(0.03, 0.64, '其他全部跟 mix_exp3 一樣', fontsize=13.5, fontweight='bold', color=TEAL, va='top')
same = ['資料：同一批 train 418 / val 51 / test 51', '層數：一樣縮小 4 次、放大 4 次',
        '版本：位移場版（--int-steps 0）', '平滑權重：λ = 1.0', '損失：NCC（9×9×9 窗格）',
        '訓練：250 epoch、學習率 1e-4']
for i, t in enumerate(same):
    ax.text(0.05, 0.575 - i * 0.063, '・' + t, fontsize=11.5, va='top')

ax.text(0.03, 0.17, '在問什麼', fontsize=13.5, fontweight='bold', va='top')
ax.text(0.03, 0.115, '模型是不是太小、裝不下？\nDice 跟著上去 → 是；沒動 → 瓶頸在別處',
        fontsize=11.5, va='top', color=INK)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet_explained.png')
fig.savefig(out, dpi=115, facecolor=PAPER)
print('->', out)
