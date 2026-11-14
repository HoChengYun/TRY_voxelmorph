# -*- coding: utf-8 -*-
"""Part 3：架構修改路線上五個模型之形變場（VNT045）——形變後影像、差值圖、變形網格、位移場 RGB
→ models\\deck_charts\\1111_p3_deform_cor.png、1111_p3_deform_sag.png、1111_p3_deform.json

畫法、切面、刻度同 make_p2_deform.py（Part 2 之圖）。推論同 ASD/test_dice.py：model(受試者, atlas, registration=True) → moved, flow；
標籤以最近鄰搬，VNT045 之 Dice 與 test CSV 比對（確認推論正確）。第一次跑時把兩刀存成 1111_p3_deform_cache_VNT045.npz，之後不再推論。
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import binary_fill_holes

from style import J, OUT, INK, MUTED, PAPER, W, save

S = 'VNT045'
# key：(顯示名稱, 顏色, models 子資料夾, epoch)；顏色同 make_p3.py
ROUTE = {'vxm': ('① VoxelMorph', '#4B5D73', 'mix_exp3', '0240'),
         'svf': ('② SVF', '#D08C2B', 'mix_exp6', '0190'),
         'casc': ('③ ＋Cascade', '#2F7FD0', 'mix_cascade_rcn', '0195'),
         'pyr': ('③ ＋C2F', '#0E7C7B', 'mix_pyramid_dwp', '0205'),
         'comb': ('④ Cascade × C2F', '#7A4FA0', 'mix_cascade_pyramid_rcn_dwp', '0155')}
RGB_MM = 8.0
atlas = np.load(J('IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
aseg = np.load(J('IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg'].astype(np.int32)
FG = atlas > 0
d = np.load(J('data', 'mixed_preprocessed_v2', 'test', S + '.npz'))
vol, seg = d['vol'].astype(np.float32), d['seg'].astype(np.int32)
y0 = int(round(np.argwhere(np.isin(aseg, [10, 49])).mean(0)[1]))
xs = np.nonzero(atlas.max(axis=(1, 2)) > 0)[0]
x0 = int(round((xs.min() + xs.max()) / 2)) - 10
CUTS = {'cor': lambda a: a[:, y0, :], 'sag': lambda a: a[x0, :, :]}
AX = {'cor': (0, 2), 'sag': (1, 2)}

cpath = os.path.join(OUT, '1111_p3_deform_cache_%s.npz' % S)
if not os.path.exists(cpath):
    os.environ.setdefault('NEURITE_BACKEND', 'pytorch')
    os.environ.setdefault('VXM_BACKEND', 'pytorch')
    sys.path.insert(0, J('ASD'))
    import torch
    import voxelmorph as vxm
    from arch import load_model
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    LABELS = [int(x) for x in np.load(J('voxelmorph-code', 'data', 'labels.npz'))['labels']]
    at = torch.from_numpy(atlas)[None, None].to(dev)
    v = torch.from_numpy(vol)[None, None].to(dev)
    s_t = torch.from_numpy(seg.astype(np.float32))[None, None].to(dev)
    warp_nn = vxm.torch.layers.SpatialTransformer(atlas.shape, mode='nearest').to(dev)
    C = {}
    for k, (nm, col, n, ep) in ROUTE.items():
        m = load_model(J('models', n, ep + '.pt'), dev).to(dev).eval()
        with torch.no_grad():
            mv, fl = m(v, at, registration=True)
            sw = np.round(warp_nn(s_t, fl.float())[0, 0].cpu().numpy()).astype(np.int32)
        mv, fl = mv[0, 0].float().cpu().numpy(), fl[0].float().cpu().numpy()
        dl = []
        for lab in LABELS:
            a, b = sw == lab, aseg == lab
            dl.append(2 * (a & b).sum() / (a.sum() + b.sum()) if a.sum() + b.sum() else np.nan)
        ref = [float(r['dice_mean']) for r in csv.DictReader(open(J('models', n, 'dice_%s.csv' % ep), encoding='utf-8')) if r['file'] == S + '.npz'][0]
        print('%-5s VNT045 Dice %.4f（test CSV %.4f）' % (k, np.nanmean(dl), ref))
        assert abs(np.nanmean(dl) - ref) < 2e-3, k
        for p, f in CUTS.items():
            C['%s_%s_img' % (k, p)] = f(mv)
            C['%s_%s_u' % (k, p)] = np.stack([f(c) for c in fl])
        C[k + '_umag'] = np.array(float(np.sqrt((fl ** 2).sum(0))[FG].mean()))
        C[k + '_mae'] = np.array(float(np.abs(mv - atlas)[FG].mean()))
        del m
        torch.cuda.empty_cache()
    np.savez_compressed(cpath, **C)
C = dict(np.load(cpath))
COLS = [('moving', 'Moving（affine）', INK)] + [(k, v_[0], v_[1]) for k, v_ in ROUTE.items()]
MAE = {'moving': float(np.abs(vol - atlas)[FG].mean())}
MAE.update({k: float(C[k + '_mae']) for k in ROUTE})
UMAG = {'moving': 0.0}
UMAG.update({k: float(C[k + '_umag']) for k in ROUTE})


def crop(p):
    m = CUTS[p](FG)
    a, b = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    return slice(max(a.min() - 4, 0), a.max() + 5), slice(max(b.min() - 4, 0), b.max() + 5)


def show2(a):
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
    ax.text(0.03, 0.03, s, transform=ax.transAxes, fontsize=7.5, va='bottom', color=INK,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))


def fig_plane(p, title):
    s1, s2 = crop(p)
    A = CUTS[p](atlas)[s1, s2]
    fg = CUTS[p](FG)[s1, s2]
    filled = binary_fill_holes(show2(fg)).astype(float)
    h, w = show2(A).shape
    n = len(COLS)
    pw = W * 0.93 / (n + 1.1)
    ph = pw * h / w
    fig = plt.figure(figsize=(W, 0.6 + 4 * ph + 0.15), facecolor=PAPER)
    gs = fig.add_gridspec(4, n + 2, width_ratios=[1] * n + [0.06, 1.05], left=0.05, right=0.995,
                          top=1 - 0.5 / fig.get_size_inches()[1], bottom=0.01, wspace=0.03, hspace=0.03)
    rows = ['形變後影像', '差值（− atlas）', '變形網格', '位移場 RGB']
    for j, (k, nm, col) in enumerate(COLS):
        if k == 'moving':
            img, u = CUTS[p](vol), np.zeros((3,) + CUTS[p](vol).shape, np.float32)
        else:
            img, u = C['%s_%s_img' % (k, p)], C['%s_%s_u' % (k, p)]
        img, u = img[s1, s2], u[:, s1, s2]
        ah, av = AX[p]
        axs = [fig.add_subplot(gs[r, j]) for r in range(4)]
        for a in axs:
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        axs[0].imshow(show2(img), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axs[0].set_title(nm, fontsize=9.5, color=col, fontweight='bold', pad=3)
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
                axs[r].set_ylabel(lab, fontsize=9, color=MUTED)
    ra = fig.add_subplot(gs[0, n + 1])
    ra.imshow(show2(A), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ra.set_title('Fixed（MNI152）', fontsize=9.5, color=INK, fontweight='bold', pad=3)
    ra.set_xticks([]); ra.set_yticks([])
    rb = fig.add_subplot(gs[1, n + 1]); rb.axis('off')
    cax = rb.inset_axes([0.08, 0.62, 0.84, 0.1])
    cb = fig.colorbar(ScalarMappable(Normalize(-0.5, 0.5), 'RdBu_r'), cax=cax, orientation='horizontal', ticks=[-0.5, 0, 0.5])
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    rb.text(0.5, 0.28, '形變後 − atlas\n左下：腦內 MAE', ha='center', va='center', fontsize=7.5, color=MUTED, transform=rb.transAxes)
    rc = fig.add_subplot(gs[2, n + 1]); rc.axis('off')
    rc.text(0.5, 0.5, 'atlas 上每 5 mm\n一條線，經 x + u(x)\n映射；左下：腦內\n平均位移', ha='center', va='center', fontsize=7.5,
            color=MUTED, transform=rc.transAxes)
    rd = fig.add_subplot(gs[3, n + 1]); rd.axis('off')
    for m_, (colr, t) in enumerate([((1, 0.5, 0.5), 'R：左右'), ((0.5, 1, 0.5), 'G：前後'), ((0.5, 0.5, 1), 'B：上下')]):
        yy = 0.85 - m_ * 0.22
        rd.add_patch(plt.Rectangle((0.1, yy - 0.08), 0.14, 0.16, color=colr, transform=rd.transAxes, clip_on=False))
        rd.text(0.3, yy, t, va='center', fontsize=7.5, color=INK, transform=rd.transAxes)
    rd.text(0.1, 0.12, '±%g mm 飽和' % RGB_MM, va='center', fontsize=7, color=MUTED, transform=rd.transAxes)
    fig.text(0.05, 1 - 0.2 / fig.get_size_inches()[1], title, fontsize=9.5, color=MUTED, va='center')
    return fig


f = fig_plane('cor', '%s，冠狀面第 %d 張（經過 atlas 視丘中心）；同 Part 2 之畫法，各方法同一刻度' % (S, y0))
save(f, '1111_p3_deform_cor.png')
f = fig_plane('sag', '%s，矢狀面第 %d 張（正中線往受試者左側 10 張，經過小腦）；各方法同一刻度' % (S, x0))
save(f, '1111_p3_deform_sag.png')
json.dump({'mae': MAE, 'umag': UMAG}, open(os.path.join(OUT, '1111_p3_deform.json'), 'w'), indent=1)
print({k: (round(MAE[k], 4), round(UMAG[k], 2)) for k in MAE})
