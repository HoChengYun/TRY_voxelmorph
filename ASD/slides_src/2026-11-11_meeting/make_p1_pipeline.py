# -*- coding: utf-8 -*-
"""Part 1：三路去頭殼（FreeSurfer／tigerbx／SynthStrip）由原始影像逐步推到訓練資料（10/29 紅字 p8）
→ models\\deck_charts\\1111_p1_pipeline.png

同一位受試者（預設 VNT045）、同一個冠狀面（經過視丘）：
  ① rawavg（conform 前）② orig.mgz（conform）③ 腦遮罩 ④ orig ⊙ 遮罩 ⑤ N4 ⑥ affine 對到 MNI152 ⑦ 正規化＋aseg
①～④ 是 FreeSurfer／tigerbx 兩端交的示範檔；⑤～⑦ 是 p1_demo_preproc.py 照 preprocess_fs.py 跑出來的（v3 尚未全批執行）。
"""
import os
import sys

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from style import J, OUT, INK, MUTED, RULE, W, save, load_lut, colorize as _colorize

LUT = load_lut()
colorize = lambda L: _colorize(L, LUT)     # noqa: E731

S = sys.argv[1] if len(sys.argv) > 1 else 'VNT045'
D = J('data', 'example_for_meeting', S)
ARMS = [('freesurfer', 'FreeSurfer', 'recon-all 7.4.1\nbrainmask.mgz', 'freesurfer_mask', '#D62728'),
        ('tigerbx', 'tigerbx', 'tigerbx 0.2.3\nBET', 'tigerbx_mask', '#2F7FD0'),
        ('synthstrip', 'SynthStrip', 'mri_synthstrip\n（FreeSurfer 7.4.1）', 'synthstrip_mask', '#3E9B3A')]

oimg = nib.load(os.path.join(D, 'orig.nii.gz'))
canon = lambda arr, aff: np.asarray(nib.as_closest_canonical(nib.Nifti1Image(arr, aff)).dataobj)     # noqa: E731
orig = canon(np.asarray(oimg.dataobj).astype(np.float32), oimg.affine)
seg_c = canon(np.asarray(nib.load(os.path.join(D, 'labels', 'aseg.nii.gz')).dataobj).astype(np.int16), oimg.affine)
ctr = np.argwhere(np.isin(seg_c, [10, 49])).mean(0)                     # 視丘中心（canonical 格子）
yi = int(round(ctr[1]))
oc = nib.as_closest_canonical(oimg)
world = oc.affine @ np.r_[ctr, 1]
raw_i = nib.as_closest_canonical(nib.load(os.path.join(D, 'raw.nii.gz')))
raw = np.asarray(raw_i.dataobj).astype(np.float32)
ry = int(round((np.linalg.inv(raw_i.affine) @ world)[1]))               # 同一個物理位置在 rawavg 的冠狀面
raw_shape = nib.load(os.path.join(D, 'raw.nii.gz')).shape
aseg_atlas = np.load(J('IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg']
y0 = int(round(np.argwhere(np.isin(aseg_atlas, [10, 49])).mean(0)[1]))  # atlas 格子的冠狀面（同 Part 2）

nat = lambda a: a[:, yi, :].T                                           # noqa: E731  canonical：origin lower、圖左 = 受試者左側
atl = lambda a: np.rot90(a[:, y0, :])                                   # noqa: E731  atlas 格子（同 make_p2_deform.py）
hm = nat(orig > 2)
P, Q = np.nonzero(hm.any(1))[0], np.nonzero(hm.any(0))[0]
NLIM = (Q.min() - 3, Q.max() + 3), (P.min() - 3, P.max() + 3)
rh = raw[:, ry, :].T > np.percentile(raw, 60)
P2, Q2 = np.nonzero(rh.any(1))[0], np.nonzero(rh.any(0))[0]
RLIM = (Q2.min() - 3, Q2.max() + 3), (P2.min() - 3, P2.max() + 3)

fig = plt.figure(figsize=(W, 6.9))
FW, FH = fig.get_size_inches()
pw, ph = 1.42, 1.62
x_cols = [0.1, 1.78] + [4.12 + j * 1.6 for j in range(5)]
y_rows = [0.95 + r * 1.86 for r in range(3)]                            # 從上量起（英吋）
ax_at = lambda x, y: fig.add_axes([x / FW, 1 - (y + ph) / FH, pw / FW, ph / FH])   # noqa: E731


def panel(a, img, lim=None, origin='lower', vmax=None):
    a.imshow(img, cmap='gray', origin=origin, vmin=0, vmax=vmax if vmax else np.percentile(img[img > 0], 99.5), interpolation='nearest')
    if lim:
        a.set_xlim(*lim[0])
        a.set_ylim(*lim[1])
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(RULE)


HEAD = ['① 原始影像\n（DICOM → NIfTI）', '② Conform\n（256³、1 mm）', '③ 腦遮罩 $M_k$', '④ 去頭殼影像\n$I_k = I_{orig} \\odot M_k$',
        '⑤ N4 偏場校正\n（ANTs）', '⑥ Affine 配準\n至 MNI152（ANTs）', '⑦ 強度正規化 [0, 1]\n＋aseg（同一變換）']
mid = y_rows[1]
a = ax_at(x_cols[0], mid); panel(a, raw[:, ry, :].T, RLIM)
a.text(0.5, -0.04, '%d×%d×%d' % raw_shape, transform=a.transAxes, ha='center', va='top', fontsize=8, color=MUTED)
a = ax_at(x_cols[1], mid); panel(a, nat(orig), NLIM)
a.text(0.5, -0.04, '256×256×256', transform=a.transAxes, ha='center', va='top', fontsize=8, color=MUTED)
for j, h in enumerate(HEAD):
    fig.text((x_cols[j] + pw / 2) / FW, 1 - 0.42 / FH, h, ha='center', va='center', fontsize=9, color=INK, linespacing=1.25)
for r, (key, name, tool, mfile, col) in enumerate(ARMS):
    y = y_rows[r]
    mk = canon((np.asarray(nib.load(os.path.join(D, mfile + '.nii.gz')).dataobj) > 0).astype(np.uint8), oimg.affine) > 0
    a = ax_at(x_cols[2], y); panel(a, nat(orig), NLIM)
    ov = np.zeros(nat(mk).shape + (4,))
    ov[nat(mk)] = list(plt.matplotlib.colors.to_rgba(col, 0.35))
    a.imshow(ov, origin='lower', interpolation='nearest')
    a.contour(nat(mk).astype(float), levels=[0.5], colors=col, linewidths=1.0)
    a.set_ylabel('%s\n%s' % (name, tool), fontsize=8.5, color=col, fontweight='bold', linespacing=1.2, labelpad=4)
    a = ax_at(x_cols[3], y); panel(a, nat(orig * mk), NLIM)
    Z = np.load(os.path.join(OUT, '1111_p1_demo_%s_%s.npz' % (S, key)))
    a = ax_at(x_cols[4], y); panel(a, nat(canon(Z['n4'], oimg.affine)), NLIM)
    a = ax_at(x_cols[5], y); panel(a, atl(Z['reg']), origin='upper')
    a = ax_at(x_cols[6], y); panel(a, atl(Z['vol']), origin='upper', vmax=1)
    L = atl(Z['seg'])
    a.imshow(np.concatenate([colorize(L), (L > 0)[..., None] * 0.45], -1), origin='upper', interpolation='nearest')
    if r == 2:
        for j, t in ((5, '192×224×192'), (6, 'npz：vol、seg')):
            fig.text((x_cols[j] + pw / 2) / FW, 1 - (y + ph + 0.12) / FH, t, ha='center', va='top', fontsize=8, color=MUTED)
# 箭頭：①→②、②→三路的③、各列 ③→④→…→⑦
arr = lambda x1, y1, x2, y2: fig.add_artist(FancyArrowPatch((x1 / FW, 1 - y1 / FH), (x2 / FW, 1 - y2 / FH), transform=fig.transFigure,  # noqa: E731
                                                            arrowstyle='-|>', mutation_scale=9, color=MUTED, lw=0.9))
arr(x_cols[0] + pw + 0.02, mid + ph / 2, x_cols[1] - 0.02, mid + ph / 2)
for y in y_rows:
    arr(x_cols[1] + pw + 0.02, mid + ph / 2, x_cols[1] + pw + 0.3, y + ph / 2)
    for j in range(2, 6):
        arr(x_cols[j] + pw + 0.02, y + ph / 2, x_cols[j + 1] - 0.02, y + ph / 2)
# 誰做的：虛擬機 C（FreeSurfer 端）／tigerbx 端／筆電 A（VoxelMorph 端）
yb = 6.62
for x1, x2, t in ((x_cols[0], x_cols[3] + pw, 'FreeSurfer 端（虛擬機）；tigerbx 那一列由 tigerbx 端產生'), (x_cols[4], x_cols[6] + pw, 'VoxelMorph 端（筆電，ASD\\preprocess_fs.py --n4）')):
    fig.add_artist(plt.Line2D([x1 / FW, x2 / FW], [1 - yb / FH] * 2, color=MUTED, lw=1, transform=fig.transFigure))
    fig.text((x1 + x2) / 2 / FW, 1 - (yb + 0.08) / FH, t, ha='center', va='top', fontsize=8.5, color=MUTED)
save(fig, '1111_p1_pipeline.png')
print('冠狀面：canonical y = %d、rawavg y = %d、atlas y = %d' % (yi, ry, y0))
