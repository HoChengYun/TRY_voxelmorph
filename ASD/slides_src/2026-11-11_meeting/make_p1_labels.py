# -*- coding: utf-8 -*-
"""Part 1：FreeSurfer 的 5 種標籤檔（10/29 紅字 p17）＋三種遮罩比較（p8）
→ models\\deck_charts\\1111_p1_labels.png、1111_p1_masks.png、1111_p1_labels.json

資料：data\\example_for_meeting\\<受試者>\\（FreeSurfer、tigerbx 兩邊 11-03 交的示範檔；256³、1 mm、LIA，跟 orig 同一個格子）
  標籤顏色用 FreeSurfer 官方 FreeSurferColorLUT.txt（D:\\MyHome\\MRI\\FreeSurfer\\base_v1\\）。
  SynthStrip 的遮罩是 FreeSurfer 端 11-04 用 mri_synthstrip -m 重跑取得（去頭殼影像與原交付逐體素相同）。
  殘留 = 遮罩內、但在「aseg 補成的球」外（closing 半徑 5 mm＋補洞，同 ASD\\skullstrip_volume.py 的 make_ball，10/14 紅字的算法）。
"""
import os
import json

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import J, OUT, INK, MUTED, W, save

D = J('data', 'example_for_meeting')
LUT_P = r'D:\MyHome\MRI\FreeSurfer\base_v1\FreeSurferColorLUT.txt'
LUT = {}
for line in open(LUT_P, encoding='utf-8', errors='ignore'):
    p = line.split()
    if len(p) >= 5 and p[0].isdigit():
        LUT[int(p[0])] = (int(p[2]) / 255, int(p[3]) / 255, int(p[4]) / 255)


def canon(path, dtype=None):
    """讀進來轉成 RAS（陣列軸：x 左→右、y 後→前、z 下→上）。"""
    a = np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj)
    return a.astype(dtype) if dtype else a


def colorize(lab):
    rgb = np.zeros(lab.shape + (3,))
    for v in np.unique(lab):
        if v:
            rgb[lab == v] = LUT.get(int(v), (0.5, 0.5, 0.5))
    return rgb


def make_ball(brain, r=5):
    dil = ndi.distance_transform_edt(~brain) <= r
    return ndi.binary_fill_holes((ndi.distance_transform_edt(dil) > r) | brain)


def view(a, plane, i):                                # 顯示：上 = 上（z 大），冠狀／水平的圖左 = 受試者左側
    return {'cor': a[:, i, :], 'axi': a[:, :, i], 'sag': a[i, :, :]}[plane].T


# ════ 1. 5 種標籤檔（VNT045）════
S = 'VNT045'
FILES = [('aseg', 'aseg', '皮質下分割\n皮質左右各 1 塊'), ('aparc+aseg', 'aparc+aseg', 'Desikan–Killiany\n皮質每半球 34 區'),
         ('aparc.DKTatlas+aseg', 'aparc.DKTatlas+aseg', 'DKT\n皮質每半球 31 區'), ('aparc.a2009s+aseg', 'aparc.a2009s+aseg', 'Destrieux\n皮質每半球 74 區'),
         ('wmparc', 'wmparc', '白質依最近之\n皮質分區切分')]
orig = canon(os.path.join(D, S, 'orig.nii.gz'), np.float32) / 255
labs = {k: canon(os.path.join(D, S, 'labels', f + '.nii.gz'), np.int32) for k, f, _ in FILES}
info = {}
for k, _, _ in FILES:
    u = np.unique(labs[k])
    u = u[u > 0]
    info[k] = dict(n=int(len(u)), ctx_lh=int(((u >= 1000) & (u < 1100)).sum() + ((u >= 11100) & (u < 11200)).sum()),
                   wm_lh=int(((u >= 3000) & (u < 3100)).sum()), unseg_wm=int(np.isin(u, [5001, 5002]).sum()))
th = np.argwhere(np.isin(labs['aseg'], [10, 49])).mean(0).round().astype(int)
vent = np.argwhere(np.isin(labs['aseg'], [4, 43])).mean(0).round().astype(int)
planes = [('cor', th[1], '冠狀面（經過視丘）'), ('axi', vent[2] + 8, '水平面（側腦室上方）')]
bm = canon(os.path.join(D, S, 'freesurfer_mask.nii.gz')) > 0
fig, ax = plt.subplots(2, 5, figsize=(W, 6.3))
fig.subplots_adjust(left=0.04, right=0.995, top=0.83, bottom=0.12, wspace=0.03, hspace=0.04)
for r, (p, i, pn) in enumerate(planes):
    m = view(bm, p, i)
    P, Q = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    for c, (k, f, desc) in enumerate(FILES):
        a = ax[r, c]
        a.imshow(view(orig, p, i), cmap='gray', vmin=0, vmax=0.6, origin='lower', interpolation='nearest')
        L = view(labs[k], p, i)
        rgba = np.concatenate([colorize(L), (L > 0)[..., None] * 0.85], -1)
        a.imshow(rgba, origin='lower', interpolation='nearest')
        a.set_xlim(Q.min() - 4, Q.max() + 4)
        a.set_ylim(P.min() - 4, P.max() + 4)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if r == 0:
            a.set_title('%s\n%s' % (k, desc), fontsize=9.5, color=INK, linespacing=1.25)
        if c == 0:
            a.set_ylabel(pn, fontsize=9.5, color=MUTED)
    for c, (k, _, _) in enumerate(FILES):
        pass
for c, (k, _, _) in enumerate(FILES):
    s = '%d 個標籤' % info[k]['n']
    if info[k]['wm_lh']:
        s += '（白質每半球 %d 區）' % info[k]['wm_lh']
    ax[1, c].text(0.5, -0.06, s, transform=ax[1, c].transAxes, ha='center', va='top', fontsize=9, color=INK)
fig.text(0.04, 0.95, '%s 之 FreeSurfer 標籤檔（recon-all 7.4.1，256³、1 mm）；顏色為 FreeSurferColorLUT；皮質下結構五種檔案逐體素相同，差別在皮質與白質之切分' % S,
         fontsize=9.5, color=MUTED)
save(fig, '1111_p1_labels.png')

# ════ 2. 三種遮罩比較（三位示範受試者）════
ARMS = [('FreeSurfer', 'freesurfer_mask', '#D62728'), ('tigerbx', 'tigerbx_mask', '#2F7FD0'), ('SynthStrip', 'synthstrip_mask', '#3E9B3A')]
SUBJ = ['VNT045', 'sub-0043', 'D031']
vols = {}
fig, ax = plt.subplots(2, 3, figsize=(W, 7.6))
fig.subplots_adjust(left=0.04, right=0.995, top=0.88, bottom=0.11, wspace=0.03, hspace=0.06)
for c, s in enumerate(SUBJ):
    o = canon(os.path.join(D, s, 'orig.nii.gz'), np.float32) / 255
    seg = canon(os.path.join(D, s, 'labels', 'aseg.nii.gz'), np.int32)
    ball = make_ball(seg > 0)
    masks = {n: canon(os.path.join(D, s, f + '.nii.gz')) > 0 for n, f, _ in ARMS}
    vols[s] = {n: dict(mask_cm3=float(m.sum() / 1000), residue_cm3=float((m & ~ball).sum() / 1000)) for n, m in masks.items()}
    ctr = np.argwhere(seg > 0).mean(0).round().astype(int)
    for r, (p, i, pn) in enumerate((('cor', ctr[1], '冠狀面'), ('sag', ctr[0] - 25, '矢狀面'))):
        a = ax[r, c]
        a.imshow(view(o, p, i), cmap='gray', vmin=0, vmax=0.6, origin='lower', interpolation='nearest')
        a.contour(view(ball, p, i).astype(float), levels=[0.5], colors='white', linewidths=0.9, linestyles='dashed')
        for n, f, col in ARMS:
            a.contour(view(masks[n], p, i).astype(float), levels=[0.5], colors=col, linewidths=1.1)
        hm = view(o > 0.02, p, i)
        P, Q = np.nonzero(hm.any(1))[0], np.nonzero(hm.any(0))[0]
        a.set_xlim(Q.min(), Q.max())
        a.set_ylim(P.min(), P.max())
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if r == 0:
            a.set_title(s, fontsize=10.5, color=INK, fontweight='bold')
        if c == 0:
            a.set_ylabel(pn, fontsize=9.5, color=MUTED)
    t = '　'.join('%s %.0f（殘留 %.0f）' % (n, vols[s][n]['mask_cm3'], vols[s][n]['residue_cm3']) for n, _, _ in ARMS)
    ax[1, c].text(0.5, -0.04, '遮罩體積 cm³：\n' + t.replace('　', '\n'), transform=ax[1, c].transAxes, ha='center', va='top', fontsize=8.8,
                  color=INK, linespacing=1.3)
hd = [Line2D([], [], color=col, lw=1.6, label=n + ' 遮罩') for n, _, col in ARMS]
hd.append(Line2D([], [], color=MUTED, lw=1.2, ls='--', label='aseg 補成之「球」（殘留之界線）'))
fig.legend(handles=hd, loc='upper center', ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.955))
fig.text(0.04, 0.985, '三種去頭殼遮罩疊於同一張 orig.mgz；殘留 = 遮罩內、aseg 球外之體積（closing 半徑 5 mm，同 10/14 會議建議方法）',
         fontsize=9.5, color=MUTED, va='top')
save(fig, '1111_p1_masks.png')
json.dump(dict(labels=info, masks=vols), open(os.path.join(OUT, '1111_p1_labels.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(info, ensure_ascii=False))
print(json.dumps(vols, ensure_ascii=False, indent=1))
