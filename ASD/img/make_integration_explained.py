# -*- coding: utf-8 -*-
"""教學圖：速度場版的「積分」在做什麼，為什麼它不會擠爆。

用一個玩具例子（不是真的腦）：同一個「往中間擠」的方向場 v，
    左：網路給的方向（兩個版本都一樣）
    中：位移場版  φ = Id + v     每個點照箭頭一步跳過去 → 擠太用力時點會互相穿過去（紅色＝擠爆）
    右：速度場版  φ = exp(v)     把同一個箭頭切成 128 小步、跟著走 → 點會擠在一起，但永遠不會穿過去

真的 VoxelMorph 的 128 步不是一步一步走的，是「縮放再平方」：
先把 v 除以 128 得到一小步，再讓這一小步「自己接自己」7 次：1→2→4→…→128。
這張圖為了好懂，用最直接的方法一小步一小步走（結果一樣）。

輸出到同資料夾的 integration_explained.png。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
TEAL, RUST, RED = '#0E7C7B', '#A34F1B', '#D62728'

A, S = 2.2, 0.45          # 擠的力道、範圍（A > 1 時一步跳過去一定會穿過中線）
STEPS = 128


def v(x, y):
    """往中間（左右方向）擠的方向場，只在中間一塊有作用。"""
    g = np.exp(-(x ** 2 + y ** 2) / S ** 2)
    return -A * x * g, np.zeros_like(x)


def one_shot(x, y):
    vx, vy = v(x, y)
    return x + vx, y + vy


def integrate(x, y, keep=False):
    px, py = x.copy(), y.copy()
    path = [(px.copy(), py.copy())]
    for _ in range(STEPS):
        vx, vy = v(px, py)
        px, py = px + vx / STEPS, py + vy / STEPS
        if keep:
            path.append((px.copy(), py.copy()))
    return (px, py, path) if keep else (px, py)


def det_one_shot(x, y, h=1e-4):
    fx1, fy1 = one_shot(x + h, y)
    fx0, fy0 = one_shot(x - h, y)
    gx1, gy1 = one_shot(x, y + h)
    gx0, gy0 = one_shot(x, y - h)
    a, c = (fx1 - fx0) / (2 * h), (fy1 - fy0) / (2 * h)
    b, d = (gx1 - gx0) / (2 * h), (gy1 - gy0) / (2 * h)
    return a * d - b * c


def draw_grid(ax, warp, color_det=None):
    lines = np.linspace(-1, 1, 21)
    t = np.linspace(-1, 1, 400)
    segs, cols = [], []
    for c in lines:
        for xs, ys in ((np.full_like(t, c), t), (t, np.full_like(t, c))):
            wx, wy = warp(xs, ys)
            pts = np.stack([wx, wy], 1)
            seg = np.stack([pts[:-1], pts[1:]], 1)
            segs.append(seg)
            if color_det is not None:
                dd = color_det((xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2)
                cols.append(np.where(dd <= 0, RED, INK))
            else:
                cols.append(np.full(len(seg), INK))
    segs = np.concatenate(segs)
    cols = np.concatenate(cols)
    lw = np.where(cols == RED, 2.2, 0.9)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, alpha=0.95))


def frame(ax, title, sub, color):
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#D9D9D2')
    ax.set_title(title, fontsize=16, fontweight='bold', color=color, pad=26)
    ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha='center', va='bottom', fontsize=12, color=MUTED)


fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.9), facecolor=PAPER)
for ax in axes:
    ax.set_facecolor('white')

# 左：方向場
ax = axes[0]
gx, gy = np.meshgrid(np.linspace(-0.9, 0.9, 13), np.linspace(-0.9, 0.9, 13))
vx, vy = v(gx, gy)
ax.quiver(gx, gy, vx, vy, angles='xy', scale_units='xy', scale=1, color=TEAL, width=0.006)
frame(ax, '① 網路給的方向', '箭頭＝這個點要往哪移、移多少（兩個版本都一樣）', INK)

# 中：一步跳過去
ax = axes[1]
draw_grid(ax, one_shot, color_det=det_one_shot)
for y0 in (-0.12, 0.12):
    for x0 in (-0.3, 0.3):
        x1, y1 = one_shot(np.array([x0]), np.array([y0]))
        ax.annotate('', xy=(x1[0], y1[0]), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=RUST, lw=2.2))
frame(ax, '② 位移場版：一步跳過去', 'φ = Id + u　左右兩邊的點跳過頭、互相穿過 → 紅色＝擠爆', RUST)

# 右：一小步一小步走
ax = axes[2]
draw_grid(ax, integrate)
for y0 in (-0.12, 0.12):
    for x0 in (-0.3, 0.3):
        _, _, path = integrate(np.array([x0]), np.array([y0]), keep=True)
        xs = [p[0][0] for p in path]
        ys = [p[1][0] for p in path]
        ax.plot(xs, ys, color=TEAL, lw=1.6)
        ax.plot(xs[::8], ys[::8], 'o', color=TEAL, ms=3.2)
frame(ax, '③ 速度場版：切成 128 小步跟著走', 'φ = exp(v)　點會擠在一起，但永遠不會穿過去', TEAL)

fig.subplots_adjust(left=0.01, right=0.99, top=0.83, bottom=0.03, wspace=0.06)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'integration_explained.png')
fig.savefig(out, dpi=120, facecolor=PAPER)
print('->', out)

# 順便印出兩個版本有沒有擠爆（給手冊用）
xx, yy = np.meshgrid(np.linspace(-1, 1, 401), np.linspace(-1, 1, 401))
print('位移場版 擠爆比例 %.2f%%' % (100 * (det_one_shot(xx, yy) <= 0).mean()))
