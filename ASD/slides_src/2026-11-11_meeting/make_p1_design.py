# -*- coding: utf-8 -*-
"""Part 1：實驗設計架構圖（p8）、wmparc 用於殘留分析之示範（p17）
→ models\\deck_charts\\1111_p1_design.png、1111_p1_wmparc.png、1111_p1_wmparc.json

設計（10-29 老師、10-30／11-04 使用者定）：底圖固定 orig.mgz，只換腦遮罩；三組前處理相同（N4 → 各自之 affine → 正規化）；
標籤一律 FreeSurfer aseg；三組各訓練一個設定相同之 VoxelMorph，以同一份 test、同一個 atlas 標籤評估。
wmparc 示範（sub-0043，FreeSurfer 殘留最多之示範受試者）：
  殘留 R = FreeSurfer 遮罩 ∖ B（B = aseg 補成之球，closing 5 mm＋補洞，同 10/14 會議建議方法）；
  每個殘留 voxel 歸給距離最近之 wmparc 分區（皮質 1xxx/2xxx 與其下白質 3xxx/4xxx 視為同一分區）→ 各分區之殘留體積。
"""
import os
import json

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

from style import J, OUT, INK, MUTED, PAPER, W, save, load_lut, colorize

LUT = load_lut()
NAMES = {}
for line in open(r'D:\MyHome\MRI\FreeSurfer\base_v1\FreeSurferColorLUT.txt', encoding='utf-8', errors='ignore'):
    p = line.split()
    if len(p) >= 2 and p[0].isdigit():
        NAMES[int(p[0])] = p[1]

# ════ 設計架構圖 ════
fig = plt.figure(figsize=(W, 3.9), facecolor=PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, 3.9); ax.axis('off')
ARMS = [('FreeSurfer', 'brainmask.mgz', '#D62728'), ('tigerbx', 'BET', '#2F7FD0'), ('SynthStrip', 'mri_synthstrip', '#3E9B3A')]


def node(x, y, w, h, t, col=INK, fc='white', fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec=col, lw=1.4))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.3, fontweight='bold' if bold else 'normal')


def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12, color=MUTED, lw=1.1))


node(0.1, 1.55, 1.45, 0.8, 'orig.mgz\n（共同底圖）', bold=True)
ys = [2.85, 1.75, 0.65]
for (n, tool, col), y in zip(ARMS, ys):
    node(2.0, y, 1.75, 0.6, '%s\n%s' % (n, tool), col=col)
    node(4.15, y, 1.55, 0.6, r'$I_k=I_{orig}\odot M_k$', col=col)
    node(6.1, y, 2.25, 0.6, 'N4 → affine（各自）\n→ 強度正規化', col=MUTED)
    node(8.75, y, 1.5, 0.6, 'VoxelMorph\n（原論文設定）', col=col)
    arr(1.57, 1.95, 1.98, y + 0.3)
    arr(3.77, y + 0.3, 4.13, y + 0.3)
    arr(5.72, y + 0.3, 6.08, y + 0.3)
    arr(8.37, y + 0.3, 8.73, y + 0.3)
    arr(10.27, y + 0.3, 10.63, 1.95)
node(10.65, 1.4, 1.35, 1.1, 'Dice\n（FreeSurfer\naseg，同一份\ntest）', fs=9.5, bold=True)
ax.text(2.88, 3.6, '只換遮罩', ha='center', fontsize=9.5, color=MUTED)
ax.text(7.22, 3.6, '三組相同之前處理', ha='center', fontsize=9.5, color=MUTED)
ax.text(9.5, 3.6, '各訓練一個模型', ha='center', fontsize=9.5, color=MUTED)
ax.text(6.0, 0.18, '固定：底圖（orig.mgz）、標籤（FreeSurfer aseg）、切分、atlas、模型設定、評估；變因：腦遮罩 $M_k$', ha='center', fontsize=9.5, color=MUTED)
save(fig, '1111_p1_design.png')

# ════ wmparc 示範（sub-0043）════
S = 'sub-0043'
D = J('data', 'example_for_meeting', S)
canon = lambda f, dt=None: (lambda a: a.astype(dt) if dt else a)(np.asarray(nib.as_closest_canonical(nib.load(os.path.join(D, f))).dataobj))  # noqa: E731
orig = canon('orig.nii.gz', np.float32) / 255
wm = canon(os.path.join('labels', 'wmparc.nii.gz'), np.int32)
aseg = canon(os.path.join('labels', 'aseg.nii.gz'), np.int32)
fsm = canon('freesurfer_mask.nii.gz') > 0
dil = ndi.distance_transform_edt(~(aseg > 0)) <= 5
ball = ndi.binary_fill_holes((ndi.distance_transform_edt(dil) > 5) | (aseg > 0))
res = fsm & ~ball
# 分區：皮質 1000+i／2000+i 與白質 3000+i／4000+i 視為同一分區 → 以皮質之編號代表（顏色、名稱用皮質那個）
# 其他標籤（小腦、腦幹、腦室…）照 wmparc 原編號，殘留靠近小腦時就歸給小腦，不會被硬塞給最近的皮質分區
reg = wm.copy()
for lo, hi in ((3000, 3100), (4000, 4100)):
    m = (wm > lo) & (wm < hi)
    reg[m] = wm[m] - 2000
_, ind = ndi.distance_transform_edt(reg == 0, return_indices=True)
near = reg[tuple(ind)]
rnear = np.where(res, near, 0)
vols = {int(k): int((rnear == k).sum()) for k in np.unique(rnear) if k}
top = sorted(vols.items(), key=lambda kv: -kv[1])[:6]
tot = int(res.sum())
info = dict(subject=S, residue_cm3=tot / 1000, n_regions=len(vols),
            top=[dict(label=k, name=NAMES.get(k, str(k)), cm3=v / 1000, pct=100 * v / tot) for k, v in top])
json.dump(info, open(os.path.join(OUT, '1111_p1_wmparc.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(info, ensure_ascii=False, indent=1))

yi = int(np.argwhere(aseg > 0).mean(0)[1])                                  # 冠狀面：經過腦之中心
xi = int(np.argwhere(aseg > 0).mean(0)[0]) - 25
view = lambda a, p, i: {'cor': a[:, i, :], 'sag': a[i, :, :]}[p].T            # noqa: E731
fig, axs = plt.subplots(2, 2, figsize=(W * 0.62, 6.0))
fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.07, wspace=0.03, hspace=0.06)
for r, (p, i, pn) in enumerate((('cor', yi, '冠狀面'), ('sag', xi, '矢狀面'))):
    hm = view(fsm, p, i)                                                       # 以腦遮罩之範圍裁切（背景有雜訊，不能用 orig > 0）
    P, Q = np.nonzero(hm.any(1))[0], np.nonzero(hm.any(0))[0]
    zc = P.min() + int(0.4 * (P.max() - P.min()))
    P = np.r_[P, P.max() + 14]
    Q = np.r_[Q.min() - 12, Q, Q.max() + 12]
    for c in range(2):
        a = axs[r, c]
        a.imshow(view(orig, p, i), cmap='gray', vmin=0, vmax=0.6, origin='lower', interpolation='nearest')
        if c == 0:
            L = view(wm, p, i)
            a.imshow(np.concatenate([colorize(L, LUT), (L > 0)[..., None] * 0.7], -1), origin='lower', interpolation='nearest')
            R = view(res, p, i)
            ov = np.zeros(R.shape + (4,)); ov[R] = (0.84, 0.15, 0.16, 0.85)
            a.imshow(ov, origin='lower', interpolation='nearest')
        else:
            L = view(rnear, p, i)
            a.imshow(np.concatenate([colorize(L, LUT), (L > 0)[..., None] * 0.95], -1), origin='lower', interpolation='nearest')
            a.contour(view(ball, p, i).astype(float), levels=[0.5], colors='white', linewidths=0.8, linestyles='dashed')
        a.set_xlim(Q.min() - 2, Q.max() + 2)
        a.set_ylim(zc, P.max() + 2)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if r == 0:
            a.set_title(['wmparc ＋ 殘留（紅）', '殘留依最近之分區著色'][c], fontsize=10.5, color=INK)
        if c == 0:
            a.set_ylabel(pn, fontsize=10, color=MUTED)
save(fig, '1111_p1_wmparc.png')
