# -*- coding: utf-8 -*-
"""10/14 簡報「補充評估指標」的定義頁（2026-10-07）：示意圖＋編號公式 (1)～(3)。
-> models/deck_charts/1014_metric_demo.png（示意圖，合成的 2D 例子，不是本專案資料）
   models/deck_charts/1014_metric_eq.png  （公式區塊，樣式同 make_method.py／make_arch.py）

使用者：「HD95 與 SDlogJ 還沒到很了解，你可以教我」→「把 HD95 和 SDlogJ 加進去，然後更新這次 meeting 簡報」。
實際的數值在 ASD/test_dice.py --surface（surface_<epoch>.csv），這支只畫定義。
式 (3) 的分母是 atlas 非背景 voxel（論文文字也是 non-background）；但論文 Table I 標題寫明分母是 5.2 M voxel，
我們只有 1.87 M，百分比不能直接比 → 跟論文比較用 folding voxel 數（使用者 2026-10-07 決定）。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
W, DPI = 12.13, 200
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
TEAL, RUST, BLUE, RED = '#0E7C7B', '#A34F1B', '#2F7FD0', '#C0392B'


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


# ── 示意圖：HD95（兩個邊界、距離分布）＋ SDlogJ（平滑 vs 劇烈的 log|J|）──────────────
N = 100
yy, xx = np.mgrid[:N, :N]
A = (xx - 50) ** 2 + (yy - 50) ** 2 <= 25 ** 2
ang = np.arctan2(yy - 50, xx - 52)
B = (xx - 52) ** 2 + (yy - 50) ** 2 <= (25 + 1.5 * np.sin(3 * ang)) ** 2
B |= (yy == 50) & (xx >= 76) & (xx <= 85)                                  # 少數 voxel 錯得很遠（突刺，< 5% 的邊界點）
bA, bB = A & ~ndimage.binary_erosion(A), B & ~ndimage.binary_erosion(B)
toA, toB = ndimage.distance_transform_edt(~bA), ndimage.distance_transform_edt(~bB)
dAB, dBA = toB[bA], toA[bB]
d = np.concatenate([dAB, dBA])
hd95 = max(np.percentile(dAB, 95), np.percentile(dBA, 95))                  # 同 test_dice.py：兩個方向各取 P95、取較大者
hd, assd = d.max(), d.mean()
dice = 2 * (A & B).sum() / (A.sum() + B.sum())

rng = np.random.default_rng(0)


def logj(sigma, amp):
    u = [ndimage.gaussian_filter(rng.standard_normal((N, N)), sigma) for _ in range(2)]
    u = [amp * x / np.abs(x).max() for x in u]
    gy0, gx0 = np.gradient(u[0])
    gy1, gx1 = np.gradient(u[1])
    return np.log(np.clip((1 + gy0) * (1 + gx1) - gx0 * gy1, 1e-9, None))


L1, L2 = logj(10, 4), logj(4, 6)

fig = plt.figure(figsize=(W, 3.05), facecolor=PAPER)
ax = fig.add_axes([0.005, 0.03, 0.21, 0.72])
ax.imshow(np.where(A & B, 0.86, np.where(A ^ B, 0.62, 1.0)), cmap='gray', vmin=0, vmax=1, origin='lower')
ax.contour(A, levels=[0.5], colors=TEAL, linewidths=2)
ax.contour(B, levels=[0.5], colors=RUST, linewidths=2)
iy, ix = np.unravel_index(np.argmax(np.where(bB, toA, -1)), toA.shape)
pts = np.argwhere(bA)
ty, tx = pts[np.argmin(((pts - [iy, ix]) ** 2).sum(1))]
ax.annotate('', xy=(tx, ty), xytext=(ix, iy), arrowprops=dict(arrowstyle='<->', color=RED, lw=1.8))
ax.text(ix + 1, iy + 6, r'$d=%.0f$' % toA[iy, ix] + ' mm', color=RED, fontsize=11, ha='center', va='bottom',
        bbox=dict(fc='white', ec='none', pad=1))
ax.set_xlim(15, 95)
ax.set_ylim(15, 85)
ax.axis('off')

ax = fig.add_axes([0.265, 0.17, 0.235, 0.56])
ax.hist(d, bins=np.arange(0, d.max() + 1.5, 0.5), color='#B4B2A9')
top = ax.get_ylim()[1]
for v, nm, c, ls, yf in ((hd95, 'HD95', RUST, '-', 0.95), (hd, 'HD', RED, '--', 0.62)):
    ax.axvline(v, color=c, lw=2, ls=ls)
    ax.text(v + (0.3 if nm == 'HD95' else -0.3), top * yf, '%s = %.1f mm' % (nm, v), color=c, fontsize=10.5,
            fontweight='bold', va='top', ha='left' if nm == 'HD95' else 'right')
ax.set_xlabel(r'$d$' + '（mm）', fontsize=10.5)
ax.set_yticks([])
ax.tick_params(labelsize=9.5)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)

for k, (L, nm, col) in enumerate(((L1, '平滑', TEAL), (L2, '劇烈', RUST))):
    ax = fig.add_axes([0.545 + k * 0.165, 0.03, 0.15, 0.72])
    im = ax.imshow(L, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    ax.axis('off')
    ax.text(0.5, 1.03, '%s：SDlogJ = %.2f' % (nm, L.std()), transform=ax.transAxes, ha='center', va='bottom',
            fontsize=11.5, fontweight='bold', color=col)
cax = fig.add_axes([0.885, 0.12, 0.012, 0.56])
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=9.5)
cb.set_label(r'$\log|J_\phi|$', fontsize=11)
fig.text(0.945, 0.66, '+ 膨脹', fontsize=10.5, color=RED)
fig.text(0.945, 0.12, '− 收縮', fontsize=10.5, color=BLUE)

for x, t in ((0.12, '(a) 邊界 ' + r'$\partial A$' + '（atlas）與 ' + r'$\partial B$' + '（配準後）\nDice = %.3f' % dice),
             (0.383, '(b) 所有邊界點之距離 ' + r'$d$' + '\n少數錯得很遠之點只影響 HD'),
             (0.70, '(c) ' + r'$\log|J_\phi|$' + '：兩個形變場\n（folding 皆近於 0）')):
    fig.text(x, 0.98, t, ha='center', va='top', fontsize=12, fontweight='bold', color=INK, linespacing=1.35)
save(fig, '1014_metric_demo.png')
print('demo: dice %.3f hd95 %.2f hd %.1f assd %.2f sdlogj %.3f / %.3f' % (dice, hd95, hd, assd, L1.std(), L2.std()))


# ── 公式區塊（同 make_method.py：公式置中、編號靠右，下面是「其中」）──────────────────
def eq_block(name, eqs, lines, h, gap=0.31, fs=20):
    fig = plt.figure(figsize=(W, h), facecolor=PAPER)
    y = 0.36
    for eq, num in eqs:
        fig.text(0.5, 1 - y / h, eq, ha='center', va='center', fontsize=fs, color=INK)
        fig.text(0.985, 1 - y / h, r'$(%d)$' % num, ha='right', va='center', fontsize=17, color=INK)
        y += 0.62
    y -= 0.62 - 0.46
    for i, ln in enumerate(lines):
        fig.text(0.02, 1 - (y + i * gap) / h, ln, ha='left', va='top', fontsize=12.5, color=INK)
    save(fig, name)


IN, SP = '其中　', '　　　'
eq_block('1014_metric_eq.png',
         [(r'$\mathrm{HD95}(A,B)=\max\left\{P_{95}\left[d(a,\partial B)\right]_{a\in\partial A},\ '
           r'P_{95}\left[d(b,\partial A)\right]_{b\in\partial B}\right\},\qquad d(a,\partial B)=\min_{b\in\partial B}\Vert a-b\Vert$', 1),
          (r'$\mathrm{SDlogJ}=\mathrm{std}_{p\in\Omega}\ \log\max\left(|J_\phi(p)|,\ \epsilon\right),\qquad J_\phi=I+\nabla u$', 2),
          (r'$\mathrm{folding}=\frac{\left|\left\{p\in\Omega:\ |J_\phi(p)|\leq 0\right\}\right|}{|\Omega|}\times 100\%$', 3)],
         [IN + r'$\partial A$' + '、' + r'$\partial B$' + '：atlas 與配準後標籤之邊界 voxel；' + r'$P_{95}$'
          + '：第 95 百分位；每個結構各算一次，取 30 個結構之平均（1 voxel = 1 mm）',
          SP + r'$\Omega$' + '：fixed image（atlas）之非背景 voxel；' + r'$\epsilon=10^{-9}$' + '（' + r'$|J_\phi|\leq 0$'
          + ' 之處 log 無定義）；SDlogJ 越小，局部體積變化越一致',
          SP + '本研究 ' + r'$|\Omega|$' + ' ＝ 187 萬；論文 Table I 之分母為 520 萬 voxel，百分比不可直接比較，'
          + '與論文比較改用 folding voxel 數（VoxelMorph (CC)：19,077）'],
         2.95)
