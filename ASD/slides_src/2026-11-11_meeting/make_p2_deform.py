# -*- coding: utf-8 -*-
"""Part 2：形變場（VNT045）——形變後影像、差值圖（形變後 − atlas）、變形網格、位移場 RGB
→ models\\deck_charts\\1111_deform_cor.png（冠狀，經過 atlas 視丘中心）、1111_deform_sag.png（矢狀，正中線往左 10 張，切到小腦）

畫法同 10/28 簡報 make_paper_vis.py：差值圖 RdBu ±0.5、網格每 5 mm、RGB = (x, y, z) 位移、各方法同一刻度 ±8 mm。
位移場慣例都是 VoxelMorph 的 moved(x) = moving(x + u(x))，u 在 atlas 格子上、單位 voxel（= mm）。
資料：11-02 推論存的全尺寸快取（moved、seg、flow；RCN 用 rcn_torch、DWP 用原作程式，Dice 都跟 test CSV 對過），
      第一次跑時只取要畫的兩刀存成 models\\deck_charts\\1111_deform_cache_VNT045.npz，之後不再需要原快取。
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import binary_fill_holes

from style import J, OUT, MODELS, INK, MUTED, PAPER, W, save

S = 'VNT045'
OLD = r'C:\Users\h4524\AppData\Local\Temp\claude\C--Users-h4524-claude-cheng\b94b7d4d-abf6-4d59-84e4-c9799aade1ce\scratchpad'
SRC = {'vxm': 'exp3', 'rcn': 'rcn', 'dwp': 'dwp'}
RGB_MM = 8.0
atlas = np.load(J('IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
aseg = np.load(J('IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg'].astype(np.int32)
FG = atlas > 0
vol = np.load(J('data', 'mixed_preprocessed_v2', 'test', S + '.npz'))['vol'].astype(np.float32)
y0 = int(round(np.argwhere(np.isin(aseg, [10, 49])).mean(0)[1]))          # 冠狀：經過視丘中心
xs = np.nonzero(atlas.max(axis=(1, 2)) > 0)[0]
x0 = int(round((xs.min() + xs.max()) / 2)) - 10                             # 矢狀：正中線往受試者左側 10 張
CUTS = {'cor': lambda a: a[:, y0, :], 'sag': lambda a: a[x0, :, :]}
AX = {'cor': (0, 2), 'sag': (1, 2)}                                         # 平面內的兩個位移分量（橫、直）

cpath = os.path.join(OUT, '1111_deform_cache_%s.npz' % S)
if not os.path.exists(cpath):
    C = {}
    for k, t in SRC.items():
        q = np.load(os.path.join(OLD, 'cache_%s_%s.npz' % (S, t)))
        mv, fl = q['moved'].astype(np.float32), q['flow'].astype(np.float32)
        for p, f in CUTS.items():
            C['%s_%s_img' % (k, p)] = f(mv)
            C['%s_%s_u' % (k, p)] = np.stack([f(c) for c in fl])
        C[k + '_umag'] = np.array(float(np.sqrt((fl ** 2).sum(0))[FG].mean()))
        C[k + '_mae'] = np.array(float(np.abs(mv - atlas)[FG].mean()))
        print('快取', k)
    np.savez_compressed(cpath, **C)
C = dict(np.load(cpath))
COLS = [('moving', 'Moving（affine）')] + [(k, MODELS[k][0]) for k in SRC]
MAE = {'moving': float(np.abs(vol - atlas)[FG].mean())}
MAE.update({k: float(C[k + '_mae']) for k in SRC})
UMAG = {'moving': 0.0}
UMAG.update({k: float(C[k + '_umag']) for k in SRC})


def crop(p):
    m = CUTS[p](FG)
    a, b = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    return slice(max(a.min() - 4, 0), a.max() + 5), slice(max(b.min() - 4, 0), b.max() + 5)


def show2(a):                                    # 3D 的一刀（軸 i、j）→ 顯示：橫 = 第一軸、上 = z 大
    return np.rot90(a)


def grid(ax, uh, uv, step=5, lw=0.3):
    dh, dv = show2(uh), -show2(uv)
    H, Wd = dh.shape
    ii, jj = np.mgrid[0:H, 0:Wd]
    X, Y = jj + dh, ii + dv
    for i in range(0, H, step):
        ax.plot(X[i], Y[i], color=INK, lw=lw)
    for j in range(0, Wd, step):
        ax.plot(X[:, j], Y[:, j], color=INK, lw=lw)
    ax.set_xlim(-0.5, Wd - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def tag(ax, s):
    ax.text(0.03, 0.03, s, transform=ax.transAxes, fontsize=8, va='bottom', color=INK,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))


def fig_plane(p, title):
    s1, s2 = crop(p)
    A = CUTS[p](atlas)[s1, s2]
    fg = CUTS[p](FG)[s1, s2]
    filled = binary_fill_holes(show2(fg)).astype(float)
    h, w = show2(A).shape
    pw = 2.15
    ph = pw * h / w
    fig = plt.figure(figsize=(W, 0.75 + 4 * ph + 0.3), facecolor=PAPER)
    gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 0.08, 1.05], left=0.06, right=0.995, top=1 - 0.55 / fig.get_size_inches()[1],
                          bottom=0.01, wspace=0.04, hspace=0.04)
    rows = ['形變後影像', '差值（− atlas）', '變形網格', '位移場 RGB']
    for j, (k, nm) in enumerate(COLS):
        if k == 'moving':
            img, u = vol, np.zeros((3,) + CUTS[p](vol).shape, np.float32)
            img = CUTS[p](img)
        else:
            img, u = C['%s_%s_img' % (k, p)], C['%s_%s_u' % (k, p)]
        img, u = img[s1, s2], u[:, s1, s2]
        ah, av = AX[p]
        axs = [fig.add_subplot(gs[r, j]) for r in range(4)]
        for a in axs:
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        axs[0].imshow(show2(img), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axs[0].set_title(nm, fontsize=10, color=MODELS[k][2] if k in MODELS else INK, fontweight='bold', pad=3)
        axs[1].imshow(show2(img - A), cmap='RdBu_r', vmin=-0.5, vmax=0.5, interpolation='nearest')
        tag(axs[1], 'MAE %.3f' % MAE[k])
        axs[2].imshow(np.where(show2(fg), 0.9, 1.0), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        grid(axs[2], u[ah], u[av])
        tag(axs[2], '|u| %.2f mm' % UMAG[k])
        rgb = np.clip(0.5 + np.stack([show2(u[c]) for c in range(3)], -1) / (2 * RGB_MM), 0, 1)
        axs[3].imshow(rgb, interpolation='nearest')
        axs[3].contour(filled, levels=[0.5], colors='white', linewidths=0.5, alpha=0.8)
        if j == 0:
            for r, lab in enumerate(rows):
                axs[r].set_ylabel(lab, fontsize=9.5, color=MUTED)
    # 右側說明欄：atlas＋三種圖例
    ra = fig.add_subplot(gs[0, 5])
    ra.imshow(show2(A), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ra.set_title('Fixed（MNI152）', fontsize=10, color=INK, fontweight='bold', pad=3)
    ra.set_xticks([]); ra.set_yticks([])
    rb = fig.add_subplot(gs[1, 5]); rb.axis('off')
    cax = rb.inset_axes([0.08, 0.62, 0.84, 0.1])
    cb = fig.colorbar(ScalarMappable(Normalize(-0.5, 0.5), 'RdBu_r'), cax=cax, orientation='horizontal', ticks=[-0.5, 0, 0.5])
    cb.ax.tick_params(labelsize=7.5, length=2, pad=1)
    rb.text(0.5, 0.3, '形變後 − atlas（影像值 0～1）\n左下：腦內 MAE', ha='center', va='center', fontsize=8, color=MUTED, transform=rb.transAxes)
    rc = fig.add_subplot(gs[2, 5]); rc.axis('off')
    rc.text(0.5, 0.5, 'atlas 上每 5 mm 一條線\n經 x + u(x) 映射（平面分量）\n左下：腦內平均位移', ha='center', va='center', fontsize=8,
            color=MUTED, transform=rc.transAxes)
    rd = fig.add_subplot(gs[3, 5]); rd.axis('off')
    for n, (col, t) in enumerate([((1, 0.5, 0.5), 'R：左右（x）'), ((0.5, 1, 0.5), 'G：前後（y）'), ((0.5, 0.5, 1), 'B：上下（z）')]):
        yy = 0.85 - n * 0.2
        rd.add_patch(plt.Rectangle((0.08, yy - 0.07), 0.12, 0.14, color=col, transform=rd.transAxes, clip_on=False))
        rd.text(0.25, yy, t, va='center', fontsize=8, color=INK, transform=rd.transAxes)
    rd.text(0.08, 0.15, '分量 ±%g mm 飽和，0 為中間灰' % RGB_MM, va='center', fontsize=7.5, color=MUTED, transform=rd.transAxes)
    fig.text(0.06, 1 - 0.22 / fig.get_size_inches()[1], title, fontsize=10, color=MUTED, va='center')
    return fig


f = fig_plane('cor', '%s，冠狀面第 %d 張（經過 atlas 視丘中心）；差值圖、變形網格同 DWP（Jian et al.）Fig. 2，位移場 RGB 同 VoxelMorph Fig. 6' % (S, y0))
save(f, '1111_deform_cor.png')
f = fig_plane('sag', '%s，矢狀面第 %d 張（正中線往受試者左側 10 張，經過小腦）；各方法同一刻度' % (S, x0))
save(f, '1111_deform_sag.png')
json.dump({'mae': MAE, 'umag': UMAG, 'y0': y0, 'x0': x0}, open(os.path.join(OUT, '1111_deform.json'), 'w'), indent=1)
print({k: (round(MAE[k], 4), round(UMAG[k], 2)) for k in MAE})
