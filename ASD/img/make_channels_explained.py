# -*- coding: utf-8 -*-
"""手冊 §23.1 的教學圖：通道數到底是什麼、加寬 2 倍改了什麼。

拿 U-Net 第一層（縮小 1：輸入 2 張 → 預設 16 張 / 加寬 32 張特徵圖，96×112×96）當例子。
純手畫的示意圖，不讀任何資料。顏色、字型跟 make_unet_explained.py 一致。

直接執行：輸出到同資料夾的 channels_explained.png。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
INK, MUTED, PAPER, RULE = '#141A1D', '#5F6A6B', '#FAFAF8', '#C9C9C1'
TEAL, RUST = '#0E7C7B', '#A34F1B'
TEAL_L, RUST_L, GRAYF = '#DCEFEE', '#F6E7DC', '#ECECE7'

W_IN, H_IN, XMAX = 17.5, 8.2, 18.5
YMAX = H_IN * XMAX / W_IN            # 讓 x、y 一單位一樣長，方塊才是正方形

CELL, GAP = 0.26, 0.08               # 濾鏡小方塊
SLICE, STEP = 1.3, 0.045             # 特徵圖一張的邊長、疊的位移
VEC_W, VEC_H = 0.12, 0.56            # 「這一點的數字」一格


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=15,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def row(ax, yc, n, color, light, name):
    """一排：n 個濾鏡 → n 張特徵圖 → 某一點的 n 個數字。yc = 這排的中心高度。"""
    ax.text(3.0, yc + 1.25, '%s：%d 個濾鏡' % (name, n), fontsize=13.5, fontweight='bold', color=color)

    # 濾鏡：每列 8 個
    rows = n // 8
    top = yc + (rows * CELL + (rows - 1) * GAP) / 2
    for i in range(n):
        r, c = divmod(i, 8)
        ax.add_patch(Rectangle((3.0 + c * (CELL + GAP), top - CELL - r * (CELL + GAP)), CELL, CELL,
                               fc=color, ec=color, lw=0.8))
    arrow(ax, 5.8, yc, 6.3, yc)

    # 特徵圖：一張疊一張，每張一樣大
    y0 = yc - (SLICE - (n - 1) * STEP) / 2
    for k in range(n):
        ax.add_patch(Rectangle((6.4 + k * STEP, y0 - k * STEP), SLICE, SLICE, fc=light, ec=color, lw=0.7))
    fx, fy = 6.4 + (n - 1) * STEP, y0 - (n - 1) * STEP
    ax.add_patch(Ellipse((fx + SLICE / 2, fy + SLICE / 2), 0.95, 1.08, fc='none', ec=color, lw=1.0, alpha=0.55))
    ax.text(6.4 + ((n - 1) * STEP + SLICE) / 2, fy - 0.3, '%d 張特徵圖（每張一樣 96×112×96）' % n,
            ha='center', va='center', fontsize=11, color=MUTED)

    # 同一個位置穿過所有特徵圖 = 一串數字
    dx, dy = fx + 0.85, fy + 0.5
    ax.add_patch(Circle((dx, dy), 0.07, fc=color, ec=color))
    ax.plot([dx + 0.08, 9.32], [dy + 0.02, yc], color=color, lw=1.1, ls='--')
    for j in range(n):
        ax.add_patch(Rectangle((9.4 + j * VEC_W, yc - VEC_H / 2), VEC_W, VEC_H, fc=light, ec=color, lw=0.7))
    ax.text(9.4, yc + VEC_H / 2 + 0.28, '這一點 ＝ %d 個數字' % n, fontsize=12, fontweight='bold',
            color=color, va='center')


def main():
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.axis('off')

    ax.text(0.3, YMAX - 0.3, '通道數是什麼：把 U-Net 第一層（縮小 1）放大來看',
            fontsize=16, fontweight='bold', va='top', color=INK)

    # 輸入：受試者 + 模板
    ax.add_patch(Rectangle((0.40, 3.75), 1.5, 1.5, fc=GRAYF, ec=INK, lw=1.0))
    ax.add_patch(Rectangle((0.58, 3.57), 1.5, 1.5, fc=GRAYF, ec=INK, lw=1.0))
    ax.add_patch(Ellipse((1.33, 4.32), 1.05, 1.2, fc='none', ec=MUTED, lw=1.0))
    ax.text(1.3, 3.25, '受試者＋模板', ha='center', va='center', fontsize=12)
    ax.text(1.3, 2.9, '2 張，192×224×192', ha='center', va='center', fontsize=10.5, color=MUTED)
    ax.plot([2.15, 2.55], [4.3, 4.3], color=INK, lw=1.6)
    ax.plot([2.55, 2.55], [2.6, 6.0], color=INK, lw=1.6)
    arrow(ax, 2.55, 6.0, 2.9, 6.0)
    arrow(ax, 2.55, 2.6, 2.9, 2.6)

    row(ax, 6.0, 16, TEAL, TEAL_L, '預設')
    row(ax, 2.6, 32, RUST, RUST_L, '加寬')
    ax.text(3.0, 5.35, '每個濾鏡滑過整顆腦 → 產生 1 張特徵圖', fontsize=10.5, color=MUTED, va='center')
    ax.text(9.4, 5.45, '網路對「這一點周圍長什麼樣」的描述', fontsize=10.5, color=MUTED, va='center')

    # 右欄：變了什麼 / 沒變的 / 記法
    ax.plot([13.7, 13.7], [0.6, 7.7], color=RULE, lw=1.0)
    X = 14.0

    def block(y, title, color, lines, size=11.5, gap=0.42):
        ax.text(X, y, title, fontsize=13.5, fontweight='bold', color=color, va='center')
        for i, t in enumerate(lines):
            ax.text(X + 0.05, y - 0.45 - i * gap, t, fontsize=size, va='center', color=INK)

    block(7.45, '變了什麼', RUST, ['・每層特徵圖張數　× 2',
                                   '・權重　301,411 → 1,197,251（× 4）',
                                   '・顯存　7.3 → 13.5 GB（× 1.85）'])
    ax.text(X + 0.05, 5.72, '權重為什麼 × 4：每個濾鏡要讀上一層所有的圖，', fontsize=10.5, color=MUTED, va='center')
    ax.text(X + 0.05, 5.40, '上一層張數 × 2、濾鏡個數 × 2，相乘 × 4', fontsize=10.5, color=MUTED, va='center')
    block(4.75, '沒變的', TEAL, ['・每張圖的大小（解析度）',
                                 '・層數：縮小 4 次、放大 4 次',
                                 '・濾鏡邊長 3×3×3（看得多遠）',
                                 '・輸入 2 張、輸出 3 張（x、y、z 位移）'])
    block(2.35, '記法', INK, ['解析度 ＝ 看得多細',
                              '層數 ＝ 看得多遠',
                              '通道數 ＝ 每個點同時記住幾種特徵'], gap=0.40)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'channels_explained.png')
    fig.savefig(out, dpi=115, facecolor=PAPER)
    print('->', out)


if __name__ == '__main__':
    main()
