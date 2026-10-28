# -*- coding: utf-8 -*-
"""SynthSeg 之去顱骨與分割：同一位受試者跟 FreeSurfer、tigerbx 並排（10/28 簡報 ⑨，紅字以外）。

2026-10-27 使用者：「用其他工具 synthseg 來做去顱骨和分標籤」「給老師看看切出來的樣子和視覺長怎麼樣」「比 dice 就不用放了，他也還沒跑模型」
→ 只看影像與標籤，不算 Dice。SynthSeg 目前只有一位（data/synthseg/test，sub-0043，剛好也在 mixed_v2 的 test 裡）。

空間：受試者的原始空間（沒有 affine 到 atlas）。SynthSeg、tigerbx 是同一個原始掃描的格子（192×256×256、RAS、affine 完全相同）；
FreeSurfer 是 conform 後的 256³ LIA，跟原始格子只差軸的排列與整數平移，用最近鄰搬過來是逐 voxel 的重排、沒有內插（下面有 assert）。
影像：SynthSeg、tigerbx 是原始掃描強度（兩者重疊處完全相同），FreeSurfer 是 norm.mgz（偏場校正過）→ 各自除以腦內第 99.5 百分位再顯示。

  1028_ss_label.png       同上，疊上標籤（FreeSurfer LUT 顏色；CSF 改用淺藍才看得到）
  1028_ss_zoom.png        冠狀面經過海馬迴，深部結構放大
  1028_ss_volume.png      各結構體積（cm³，三套並排；大結構、小結構兩格）
  （去顱骨的對照 10-27 起併進 make_resvol.py 的 1028_resvol_slices_*.png 第三列，affine 後、跟殘留同一個算法）
  deck_data_ss.json       投影片上的數字（build.js 讀）
"""
import os
import json
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
DATA = os.path.join(ROOT, 'data')
SUBJECT = 'sub-0043'
SRC = {
    'FreeSurfer': os.path.join(DATA, 'fs_subjects_data', 'fs_for_vxm', '%s', SUBJECT + '.nii.gz'),
    'tigerbx': os.path.join(DATA, 'tigerbx_data', 'fs_for_vxm', '%s', SUBJECT + '.nii.gz'),
    'SynthSeg': os.path.join(DATA, 'synthseg', 'test', '%s', SUBJECT + '.nii.gz'),
}
TOOLS = list(SRC)
LUT_TXT = r'D:\MyHome\MRI\FreeSurfer\base_v1\FreeSurferColorLUT.txt'

ref = nib.load(SRC['SynthSeg'] % 'aseg')
VOL, SEG = {}, {}
for t, p in SRC.items():
    im, sg_ = nib.load(p % 'norm'), nib.load(p % 'aseg')
    if t == 'FreeSurfer':
        # conform 的格子 → 原始格子：必須是軸的排列＋整數平移，最近鄰才等於逐 voxel 重排
        M = np.linalg.inv(im.affine) @ ref.affine
        assert np.allclose(np.abs(M[:3, :3]).sum(0), 1) and np.allclose(M[:3, :3], np.round(M[:3, :3]), atol=1e-4) \
            and np.allclose(M[:3, 3], np.round(M[:3, 3]), atol=1e-3), 'FreeSurfer 跟原始格子不是單純的重排：\n%s' % M
        n0 = int((np.asanyarray(im.dataobj) > 0).sum())
        im, sg_ = resample_from_to(im, ref, order=0), resample_from_to(sg_, ref, order=0)
        assert int((np.asanyarray(im.dataobj) > 0).sum()) == n0, 'FreeSurfer 搬過來有 voxel 掉出格子外'
    else:
        assert np.allclose(im.affine, ref.affine) and im.shape == ref.shape
    v = np.asanyarray(im.dataobj).astype(np.float32)
    VOL[t] = np.clip(v / np.percentile(v[v > 0], 99.5), 0, 1)
    SEG[t] = np.asanyarray(sg_.dataobj).astype(np.int32)
both = (VOL['SynthSeg'] > 0) & (VOL['tigerbx'] > 0)
raw_s = np.asanyarray(nib.load(SRC['SynthSeg'] % 'norm').dataobj)
raw_t = np.asanyarray(nib.load(SRC['tigerbx'] % 'norm').dataobj)
same_intensity = bool(np.array_equal(raw_s[both], raw_t[both]))

# ── 數字 ──────────────────────────────────────────────────────────────
CSF = 24
NAME_PAIR = [  # (圖上名稱, 標籤) 左右合併；順序約略由大到小
    ('Cerebral-Cortex', (3, 42)), ('Cerebral-WM', (2, 41)), ('Cerebellum-Cortex', (8, 47)), ('Cerebellum-WM', (7, 46)),
    ('Lateral-Ventricle', (4, 43)), ('Brain-Stem', (16,)), ('Thalamus', (10, 49)), ('Putamen', (12, 51)),
    ('Hippocampus', (17, 53)), ('VentralDC', (28, 60)), ('Caudate', (11, 50)), ('Amygdala', (18, 54)),
    ('Pallidum', (13, 52)), ('4th-Ventricle', (15,)), ('3rd-Ventricle', (14,)), ('Inf-Lat-Vent', (5, 44)),
    ('Accumbens', (26, 58)), ('Choroid-plexus', (31, 63)), ('CSF', (24,)),
]
cm3 = lambda m: float(m.sum()) / 1000
labs = {t: sorted(int(x) for x in np.unique(SEG[t]) if x) for t in TOOLS}
EVAL = np.load(os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))['labels'].astype(int).tolist()
T = dict(subject=SUBJECT, same_intensity=same_intensity, tools=TOOLS,
         mask=  {t: cm3(VOL[t] > 0) for t in TOOLS},                                  # 去顱骨後非零之範圍
         tissue={t: cm3((SEG[t] > 0) & (SEG[t] != CSF)) for t in TOOLS},              # 腦組織標籤（不含 CSF）
         csf=   {t: cm3(SEG[t] == CSF) for t in TOOLS},
         n_labels={t: len(labs[t]) for t in TOOLS},
         missing={t: [l for l in labs['FreeSurfer'] if l not in labs[t]] for t in TOOLS},
         eval_missing={t: [int(l) for l in EVAL if l not in labs[t]] for t in TOOLS},
         volume={n: {t: cm3(np.isin(SEG[t], ls)) for t in TOOLS} for n, ls in NAME_PAIR})
# 去顱骨範圍內、但不在腦組織標籤裡的部分（不是前面的殘留體積：那個要先補成球，而且在 atlas 空間）
T['outside_tissue'] = {t: cm3((VOL[t] > 0) & ~((SEG[t] > 0) & (SEG[t] != CSF))) for t in TOOLS}

# ── 切面：用 FreeSurfer 的標籤找位置 ─────────────────────────────────────
cen = lambda ls, ax: int(round(np.where(np.isin(SEG['FreeSurfer'], ls))[ax].mean()))
CUT = {'cor': (1, cen((10, 49), 1), '冠狀面（經過視丘）'),
       'sag': (0, cen((16,), 0) - 2, '矢狀面（中線往左 2 voxels）'),
       'ax': (2, cen((12, 51), 2), '橫切面（經過殼核）')}
ZOOM = (1, cen((17, 53), 1))            # 冠狀面經過海馬迴
head = np.any([VOL[t] > 0 for t in TOOLS], axis=0)
lo = [max(int(np.where(head.any(axis=tuple(k for k in range(3) if k != a)))[0].min()) - 3, 0) for a in range(3)]
hi = [min(int(np.where(head.any(axis=tuple(k for k in range(3) if k != a)))[0].max()) + 4, head.shape[a]) for a in range(3)]
T['slices'] = {k: dict(axis='ijk'[a], index=i, name=n) for k, (a, i, n) in CUT.items()}


def sl2(a, ax, idx, box=None):
    """切面 → 2D（同 make_resvol.py：.T、origin lower；冠狀、橫切面影像左＝受試者左）＋裁切。"""
    c, r = min(k for k in range(3) if k != ax), max(k for k in range(3) if k != ax)
    b = box or (lo[c], hi[c], lo[r], hi[r])
    return np.take(a, idx, axis=ax).T[b[2]:b[3], b[0]:b[1]]


# ── 圖 ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
TEAL, RUST, GRAY = '#0E7C7B', '#A34F1B', '#9AA09B'
TCOL = {'FreeSurfer': GRAY, 'tigerbx': TEAL, 'SynthSeg': RUST}
CSF_RGB = (93, 173, 226)                # FreeSurfer LUT 的 CSF 是深灰（60, 60, 60），疊在影像上看不到 → 改淺藍

lut = np.zeros((256, 4))
for line in open(LUT_TXT, encoding='utf-8', errors='ignore'):
    s = line.split()
    if len(s) >= 6 and s[0].isdigit() and int(s[0]) < 256:
        lut[int(s[0])] = [int(s[2]) / 255, int(s[3]) / 255, int(s[4]) / 255, 1]          # 編號 名稱 R G B A
lut[CSF, :3] = np.array(CSF_RGB) / 255
lut[0, 3] = 0
LCMAP = ListedColormap(lut)


def save(fig, name, dpi=150):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=dpi, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


def tissue(t):
    return ((SEG[t] > 0) & (SEG[t] != CSF)).astype(float)


def grid_fig(name, draw, maxW=6.3, maxH=5.35):
    """列＝三個方向、欄＝三套。所有格子同一個比例尺（每 voxel 幾吋），欄寬取該欄最寬的方向。"""
    dims = {k: (hi[min(j for j in range(3) if j != a)] - lo[min(j for j in range(3) if j != a)],
                hi[max(j for j in range(3) if j != a)] - lo[max(j for j in range(3) if j != a)]) for k, (a, _, _) in CUT.items()}
    LM, TOP, G = 0.30, 0.30, 0.05
    cw = max(w for w, _ in dims.values())
    s = min((maxW - LM - 2 * G) / (3 * cw), (maxH - TOP - 2 * G) / sum(h for _, h in dims.values()))
    FW, H = LM + 3 * cw * s + 2 * G, TOP + sum(h for _, h in dims.values()) * s + 2 * G
    fig = plt.figure(figsize=(FW, H), facecolor=PAPER)
    y = TOP
    for k, (a, i, nm) in CUT.items():
        w, h = dims[k]
        for j, t in enumerate(TOOLS):
            x = LM + j * (cw * s + G) + (cw - w) * s / 2
            ax = fig.add_axes([x / FW, 1 - (y + h * s) / H, w * s / FW, h * s / H])
            draw(ax, t, a, i)
            ax.axis('off')
            if k == 'cor':
                fig.text((LM + j * (cw * s + G) + cw * s / 2) / FW, 1 - (TOP - 0.06) / H, t, ha='center', va='bottom',
                         fontsize=11.5, fontweight='bold', color=INK)
        fig.text((LM - 0.08) / FW, 1 - (y + h * s / 2) / H, nm.split('（')[0], ha='right', va='center', rotation=90,
                 fontsize=10.5, color=MUTED)
        y += h * s + G
    save(fig, name)


def draw_label(ax, t, a, i):
    ax.imshow(sl2(VOL[t], a, i), cmap='gray', origin='lower', vmin=0, vmax=1)
    ax.imshow(sl2(SEG[t], a, i), cmap=LCMAP, origin='lower', vmin=0, vmax=255, alpha=0.55, interpolation='nearest')


grid_fig('1028_ss_label.png', draw_label)

# 放大：冠狀面經過海馬迴，框住深部結構（尾核～海馬迴、左右殼核之外）
a, i = ZOOM
deep = np.isin(np.take(SEG['FreeSurfer'], i, axis=a), (10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 4, 43, 28, 60))
ci, ck = np.where(deep)
box = (max(ci.min() - 12, 0), min(ci.max() + 13, SEG['FreeSurfer'].shape[0]), max(ck.min() - 10, 0), min(ck.max() + 11, SEG['FreeSurfer'].shape[2]))
bw, bh = box[1] - box[0], box[3] - box[2]
FW, G, TOP = 12.13, 0.08, 0.32
s = (FW - 2 * G) / (3 * bw)
H = TOP + bh * s
fig = plt.figure(figsize=(FW, H), facecolor=PAPER)
for j, t in enumerate(TOOLS):
    ax = fig.add_axes([j * (bw * s + G) / FW, 0, bw * s / FW, bh * s / H])
    ax.imshow(sl2(VOL[t], a, i, box), cmap='gray', origin='lower', vmin=0, vmax=1)
    ax.imshow(sl2(SEG[t], a, i, box), cmap=LCMAP, origin='lower', vmin=0, vmax=255, alpha=0.45, interpolation='nearest')
    z_ = sl2(SEG[t], a, i, box)
    for l in np.unique(z_[z_ > 0]):        # 每個標籤各畫一條外框（直接對編號畫等高線，編號差很多的交界會疊成粗線）
        ax.contour((z_ == l).astype(float), levels=[0.5], colors='k', linewidths=0.35, alpha=0.7)
    ax.axis('off')
    fig.text((j * (bw * s + G) + bw * s / 2) / FW, 1 - (TOP - 0.06) / H, t, ha='center', va='bottom', fontsize=12,
             fontweight='bold', color=INK)
save(fig, '1028_ss_zoom.png')
T['zoom'] = dict(axis='j', index=int(i), mm=[int(bw), int(bh)])

# 各結構體積（cm³，三套並排）。10-27 使用者：「為啥沒有 freesurfer 的」→ 原本以 FreeSurfer = 100% 只畫兩套，改畫實際體積；
# 大小差兩個數量級，分成大結構（含 CSF）、小結構兩格
BIG = ['Cerebral-Cortex', 'Cerebral-WM', 'Cerebellum-Cortex', 'CSF', 'Cerebellum-WM', 'Lateral-Ventricle', 'Brain-Stem']
SMALL = [n for n, _ in NAME_PAIR if n not in BIG]
TCOL['FreeSurfer'] = '#6E7572'
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), facecolor=PAPER, gridspec_kw={'width_ratios': [len(BIG), len(SMALL)]})
for ax, names in zip(axes, [BIG, SMALL]):
    xs = np.arange(len(names))
    for t, dx in zip(TOOLS, (-0.27, 0, 0.27)):
        v = np.array([T['volume'][n][t] for n in names])
        ax.bar(xs + dx, v, width=0.26, color=TCOL[t], label=t, alpha=0.9)
        for x, y in zip(xs + dx, v):
            if y == 0:
                ax.text(x, ax.get_ylim()[1] * 0.02, '無此標籤', rotation=90, ha='center', va='bottom', fontsize=7, color=TCOL[t])
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=50, ha='right', fontsize=7.8)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', alpha=.3)
    ax.set_axisbelow(True)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
axes[0].set_ylabel('體積（cm³）', fontsize=9)
axes[1].legend(fontsize=8.5, frameon=False, loc='upper right')
fig.tight_layout(w_pad=0.6)
save(fig, '1028_ss_volume.png')

json.dump(T, open(os.path.join(HERE, 'deck_data_ss.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('去顱骨範圍（cm³）：', {t: round(v) for t, v in T['mask'].items()})
print('腦組織標籤（不含 CSF）：', {t: round(v) for t, v in T['tissue'].items()}, '；CSF：', {t: round(v, 1) for t, v in T['csf'].items()})
print('範圍內、腦組織以外：', {t: round(v) for t, v in T['outside_tissue'].items()})
print('SynthSeg 沒有的標籤：', T['missing']['SynthSeg'], '；評估的 30 個裡沒有：', T['eval_missing']['SynthSeg'])
print('SynthSeg 與 tigerbx 重疊處強度相同：', same_intensity)
