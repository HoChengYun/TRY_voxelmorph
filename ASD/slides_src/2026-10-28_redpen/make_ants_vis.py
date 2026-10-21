# -*- coding: utf-8 -*-
"""10/28 簡報 ⑧：同一位受試者，ANTs SyN 與 VoxelMorph 之形變結果並排 → models\\deck_charts\\1028_ants_vis.png

欄：Fixed（MNI152 atlas）、Moving（affine only）、ANTs SyN、VoxelMorph（mix_exp3，displacement field，同論文 Table I）、
    Cascade + coarse-to-fine（mix_cascade_pyramid）
列：(a) 形變後影像 m∘φ ＋ atlas 之結構輪廓（側腦室、視丘、殼核、海馬）
    (b) log|J|（atlas 空間；腦外淡化；|J| ≤ 0 以黃點標示）

ANTs 之結果讀 models\\ants_syn\\vis\\<受試者>.npz（ASD/ants_syn_baseline.py --save-dir；位移場定義在 atlas 格子上、指向受試者，
與 VoxelMorph 之 SpatialTransformer 同一慣例，--check 已驗證）；VoxelMorph 兩個模型在這裡重跑推論（GPU），結果快取於
models\\deck_charts\\1028_vis_cache_<受試者>.npz。Dice 取各模型 test CSV 之該列（正式結果），不在這裡重算。

用法
  python ASD\\slides_src\\2026-10-28_redpen\\make_ants_vis.py --subject T054
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
J = lambda *p: os.path.join(ROOT, *p)
sys.path.insert(0, J('ASD'))

ap = argparse.ArgumentParser()
ap.add_argument('--subject', default='T054')
ap.add_argument('--gpu', default='0')
args = ap.parse_args()

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
W = 12.13

atlas_vol = np.load(J('IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
atlas_seg = np.load(J('IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg'].astype(np.int32)
d = np.load(J('data', 'mixed_preprocessed_v2', 'test', args.subject + '.npz'))
vol, seg = d['vol'].astype(np.float32), d['seg'].astype(np.int32)
FG = atlas_vol > 0
S = args.subject


def jac_det(f):
    g = [[np.gradient(f[c], axis=a) for a in range(3)] for c in range(3)]
    return ((1 + g[0][0]) * ((1 + g[1][1]) * (1 + g[2][2]) - g[1][2] * g[2][1])
            - g[0][1] * (g[1][0] * (1 + g[2][2]) - g[1][2] * g[2][0])
            + g[0][2] * (g[1][0] * g[2][1] - (1 + g[1][1]) * g[2][0]))


# ── ANTs ────────────────────────────────────────────────────────────
pa = J('models', 'ants_syn', 'vis', S + '.npz')
if not os.path.exists(pa):
    sys.exit('[X] 還沒有 %s（ASD/ants_syn_baseline.py --save-dir models\\ants_syn\\vis 跑完才有）' % pa)
A = np.load(pa)
R = {'ants': {'warped': A['warped'], 'det': A['det']}}

# ── VoxelMorph（快取）───────────────────────────────────────────────────
cache = J('models', 'deck_charts', '1028_vis_cache_%s.npz' % S)
SPECS = [('mix_exp3', '0240'), ('mix_cascade_pyramid', '0140')]
if os.path.exists(cache):
    c = np.load(cache)
    for e, _ in SPECS:
        R[e] = {'warped': c[e + '_warped'], 'det': c[e + '_det']}
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != '-1' else ''
    import torch
    from arch import load_model
    dev = 'cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu'
    out = {}
    with torch.no_grad():
        for e, ep in SPECS:
            m = load_model(J('models', e, ep + '.pt'), dev).to(dev).eval()
            moved, flow = m(torch.from_numpy(vol)[None, None].to(dev), torch.from_numpy(atlas_vol)[None, None].to(dev),
                            registration=True)
            f = flow[0].cpu().numpy()
            R[e] = {'warped': moved[0, 0].cpu().numpy(), 'det': jac_det(f).astype(np.float32)}
            out[e + '_warped'], out[e + '_det'] = R[e]['warped'], R[e]['det']
            del m
            if dev == 'cuda':
                torch.cuda.empty_cache()
    np.savez_compressed(cache, **out)


def test_dice(path):
    with open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['file'] == S + '.npz':
                return float(r['dice_mean'])


DICE = {'moving': test_dice(J('models', 'mix_exp2', 'dice_baseline.csv')),
        'ants': test_dice(J('models', 'ants_syn', 'dice_syn_test.csv')),
        'mix_exp3': test_dice(J('models', 'mix_exp3', 'dice_0240.csv')),
        'mix_cascade_pyramid': test_dice(J('models', 'mix_cascade_pyramid', 'dice_0140.csv'))}

# ── 切面：冠狀面，經過 atlas 視丘的中心（RAS：軸 1 = 後→前）─────────────────────────
y = int(round(np.argwhere(np.isin(atlas_seg, [10, 49]))[:, 1].mean()))
xs, zs = np.where(FG[:, y, :])
x0, x1, z0, z1 = max(xs.min() - 4, 0), xs.max() + 5, max(zs.min() - 4, 0), zs.max() + 5
cut = lambda a: np.rot90(a[x0:x1, y, z0:z1])
STRUCT = [((4, 43), '#2F7FD0', '側腦室'), ((10, 49), '#D08C2B', '視丘'), ((12, 51), '#7A4FA0', '殼核'),
          ((17, 53), '#0E7C7B', '海馬')]

COLS = [('fixed', 'Fixed（MNI152 atlas）', None), ('moving', 'Moving（affine only）', DICE['moving']),
        ('ants', 'ANTs SyN', DICE['ants']), ('mix_exp3', 'VoxelMorph（mix_exp3）', DICE['mix_exp3']),
        ('mix_cascade_pyramid', 'Cascade + coarse-to-fine', DICE['mix_cascade_pyramid'])]
fig, axs = plt.subplots(2, 5, figsize=(W, 5.6), facecolor=PAPER)
fig.subplots_adjust(left=0.035, right=0.995, top=0.9, bottom=0.08, wspace=0.03, hspace=0.12)
for j, (k, title, dc) in enumerate(COLS):
    img = atlas_vol if k == 'fixed' else (vol if k == 'moving' else R[k]['warped'])
    a = axs[0, j]
    a.imshow(cut(img), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    for labs, colr, _ in STRUCT:
        a.contour(cut(np.isin(atlas_seg, labs)).astype(float), levels=[0.5], colors=colr, linewidths=0.9)
    a.set_title(title + ('' if dc is None else '\nDice %.3f' % dc), fontsize=10, color=INK)
    b = axs[1, j]
    if k in ('fixed', 'moving'):
        b.axis('off')
        if k == 'fixed':
            b.text(0.5, 0.5, '(b) log|J|\n（atlas 空間）\n\n紅：體積放大\n藍：體積縮小\n黃點：|J| ≤ 0\n'
                   '淺灰：atlas 值為 0\n（腦外、腦室內，不計）',
                   ha='center', va='center', fontsize=10, color=MUTED, transform=b.transAxes)
        continue
    det = R[k]['det']
    lj = np.log(np.clip(det, 1e-3, None))
    show = np.ma.masked_where(~cut(FG), cut(lj))
    b.imshow(np.full(cut(FG).shape, 0.85), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    b.imshow(show, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    fy, fx = np.where(cut(det <= 0) & cut(FG))
    b.plot(fx, fy, '.', color='#FFD400', ms=3.5, mec='#5A4A00', mew=0.3)
    nfold = int(((det <= 0) & FG).sum())
    b.text(0.02, 0.02, '腦內 |J| ≤ 0：{:,} voxels'.format(nfold), transform=b.transAxes, fontsize=9, color=INK,
           va='bottom', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
for a in axs.ravel():
    a.set_xticks([]); a.set_yticks([])
axs[0, 0].set_ylabel('(a) 形變後影像 m∘φ', fontsize=10)
handles = [Line2D([0], [0], color=c, lw=2) for _, c, _ in STRUCT]
fig.legend(handles, ['atlas ' + n for _, _, n in STRUCT], loc='lower center', ncol=4, fontsize=9.5, frameon=False,
           bbox_to_anchor=(0.5, 0.0))
fig.suptitle('%s，冠狀面（經過視丘）；輪廓為 atlas 之結構，形變後影像越貼合輪廓越好；Dice 為 30 結構之平均' % S, fontsize=10.5,
             color=MUTED, y=0.985)
p = J('models', 'deck_charts', '1028_ants_vis.png')
fig.savefig(p, dpi=130, facecolor=PAPER)
print('->', p)
