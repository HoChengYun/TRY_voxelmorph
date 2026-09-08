"""
把「Dice 到底在比什麼」畫出來。

現有的 draw-img/visualize_reg_ixi.py 是為 IXI 寫的，只看影像不看標籤。
但 Dice 比的是**結構標籤的重疊**，所以要直接把兩份標籤疊起來看：

  紅 = atlas 的結構        綠 = 受試者搬過來的結構        黃 = 兩者重疊
  黃色越多、紅綠邊緣越少 -> Dice 越高

還會畫出「配準前 vs 配準後」的對照，看模型到底改善了什麼。

用法
    python ASD\\visualize_dice.py --model models\\asd_exp1\\0190.pt --subject T023
"""

import os
import sys
import glob
import argparse
import numpy as np

os.environ.setdefault('NEURITE_BACKEND', 'pytorch')
os.environ.setdefault('VXM_BACKEND', 'pytorch')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'voxelmorph-code'))

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--subject', default=None, help='受試者 ID 或 npz 路徑；不給則取 test 第一顆')
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))
ap.add_argument('--dataset', default='ASD', help='→ data/<名稱>_preprocessed_v1/test')
ap.add_argument('--test-dir', default=None, help='預設 data/<資料集>_preprocessed_v1/test')
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--out-dir', default=None)
ap.add_argument('--gpu', default='0')
args = ap.parse_args()

# --test-dir 沒給就照 --dataset 推：data/<名稱>_preprocessed_v1/test
if args.test_dir is None:
    args.test_dir = os.path.join(ROOT, 'data', args.dataset + '_preprocessed_v1', 'test')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
import voxelmorph as vxm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# FreeSurfer LUT 名稱（只列評估用的 30 個）。左右成對的以 L/R 標示。
FS_NAME = {
    2: 'L Cerebral-WM',      41: 'R Cerebral-WM',
    3: 'L Cerebral-Cortex',  42: 'R Cerebral-Cortex',
    4: 'L Lateral-Vent',     43: 'R Lateral-Vent',
    7: 'L Cerebellum-WM',    46: 'R Cerebellum-WM',
    8: 'L Cerebellum-Ctx',   47: 'R Cerebellum-Ctx',
    10: 'L Thalamus',        49: 'R Thalamus',
    11: 'L Caudate',         50: 'R Caudate',
    12: 'L Putamen',         51: 'R Putamen',
    13: 'L Pallidum',        52: 'R Pallidum',
    17: 'L Hippocampus',     53: 'R Hippocampus',
    18: 'L Amygdala',        54: 'R Amygdala',
    28: 'L VentralDC',       60: 'R VentralDC',
    31: 'L Choroid-plexus',  63: 'R Choroid-plexus',
    14: '3rd-Ventricle',     15: '4th-Ventricle',
    16: 'Brain-Stem',        24: 'CSF',
}


# ── 找受試者 ─────────────────────────────────────────────────────────
if args.subject and os.path.exists(args.subject):
    sub_path = args.subject
elif args.subject:
    sub_path = os.path.join(args.test_dir, args.subject + '.npz')
    if not os.path.exists(sub_path):
        sys.exit('[X] 找不到 %s' % sub_path)
else:
    fs = sorted(glob.glob(os.path.join(args.test_dir, '*.npz')))
    if not fs:
        sys.exit('[X] %s 裡沒有 npz' % args.test_dir)
    sub_path = fs[0]

name = os.path.basename(sub_path)[:-4]
# 檔名沿用 draw-img/visualize_reg_ixi.py 的慣例 <類型>_<受試者>_<epoch>.png。
# 一定要含 epoch —— 否則把同一顆的不同 epoch 輸出到同一個資料夾會互相覆蓋，
# 而且不會有任何提示。
epoch = os.path.basename(os.path.abspath(args.model))
epoch = epoch[:-3] if epoch.endswith('.pt') else epoch
out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.model)),
                                       'dice_vis')
os.makedirs(out_dir, exist_ok=True)

atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
atlas_seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
LABELS = np.load(args.labels)['labels'].astype(int).tolist()
d = np.load(sub_path)
vol, seg = d['vol'].astype(np.float32), d['seg'].astype(np.int32)

print('受試者 : %s' % name)
print('模型   : %s' % args.model)

# ── 推論 ─────────────────────────────────────────────────────────────
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device).eval()
warp_nn = vxm.torch.layers.SpatialTransformer(atlas_vol.shape, mode='nearest').to(device)

with torch.no_grad():
    v = torch.from_numpy(vol)[None, None].to(device)
    a = torch.from_numpy(atlas_vol)[None, None].to(device)
    moved, flow = model(v, a, registration=True)
    moved = moved[0, 0].cpu().numpy()
    s = torch.from_numpy(seg.astype(np.float32))[None, None].to(device)
    seg_w = np.round(warp_nn(s, flow)[0, 0].cpu().numpy()).astype(np.int32)


def dice_of(x, y):
    vals = []
    for l in LABELS:
        A, B = (x == l), (y == l)
        t = A.sum() + B.sum()
        if t:
            vals.append(2.0 * (A & B).sum() / t)
    return float(np.mean(vals))


d_before = dice_of(seg, atlas_seg)
d_after = dice_of(seg_w, atlas_seg)
print('Dice  配準前 %.4f -> 配準後 %.4f  （+%.4f）' % (d_before, d_after, d_after - d_before))


def rgb_overlay(a_mask, b_mask):
    """紅=atlas、綠=受試者、黃=重疊。"""
    h, w = a_mask.shape
    img = np.zeros((h, w, 3))
    img[..., 0] = a_mask
    img[..., 1] = b_mask
    return img


D, H, W = vol.shape
# ⚠️ Sagittal 不要切正中線：那裡切不到海馬迴（距中線約 30mm），側腦室也只剩一點，
#    整欄會浪費掉。偏離 28 voxel 剛好同時切到側腦室、視丘、海馬迴。
SAG_OFFSET = 28
cuts = [(D // 2 - SAG_OFFSET, 'Sagittal (%+d mm off midline)' % -SAG_OFFSET),
        (H // 2, 'Coronal'), (W // 2, 'Axial')]


def take(arr, ax, i):
    return [arr[i], arr[:, i], arr[:, :, i]][ax].T


# ── 圖 1：標籤疊合，配準前 vs 配準後 ─────────────────────────────────
fig, ax = plt.subplots(2, 3, figsize=(14, 9.5))
for c, (i, t) in enumerate(cuts):
    A = np.isin(take(atlas_seg, c, i), LABELS).astype(float)
    for r, (sg, tag, dv) in enumerate([(seg, 'BEFORE (affine only)', d_before),
                                       (seg_w, 'AFTER (+ VoxelMorph)', d_after)]):
        B = np.isin(take(sg, c, i), LABELS).astype(float)
        ax[r][c].imshow(rgb_overlay(A, B), origin='lower',
                        interpolation='nearest', aspect='equal')
        ax[r][c].axis('off')
        if r == 0:
            ax[r][c].set_title(t, fontsize=11)
        if c == 0:
            # row 標題畫在圖的左外側，避免跟 column 標題重疊
            ax[r][c].text(-0.06, 0.5, '%s\nDice %.4f' % (tag, dv),
                          transform=ax[r][c].transAxes, rotation=90,
                          va='center', ha='center', fontsize=11, fontweight='bold')
fig.suptitle('%s — label overlap.   red = atlas,  green = subject,  yellow = agreement'
             % name, fontsize=13, fontweight='bold')
plt.tight_layout()
p1 = os.path.join(out_dir, 'labels_%s_%s.png' % (name, epoch))
plt.savefig(p1, dpi=130, bbox_inches='tight')
plt.close()
print('[OK] %s' % p1)

# ── 圖 2：影像 + 幾個代表結構的輪廓 ──────────────────────────────────
# 挑選原則：體積夠大看得到輪廓、彼此不重疊、涵蓋腦室/深部灰質/顳葉/後窩。
# 太大的（2/41 大腦白質、3/42 皮質）會蓋住整張圖，不放進來。
KEY = [
    (4, 43, 'Lateral-Ventricle', '#ff3b30'),
    (10, 49, 'Thalamus',         '#34c759'),
    (17, 53, 'Hippocampus',      '#0a84ff'),
    (11, 50, 'Caudate',          '#ffd60a'),
    (12, 51, 'Putamen',          '#ff9f0a'),
    (13, 52, 'Pallidum',         '#bf5af2'),
    (18, 54, 'Amygdala',         '#ff2d95'),
    (16, 16, 'Brain-Stem',       '#64d2ff'),
    (8, 47, 'Cerebellum-Ctx',    '#30d158'),
]
fig, ax = plt.subplots(2, 3, figsize=(14, 9.5))
for c, (i, t) in enumerate(cuts):
    bg = take(atlas_vol, c, i)
    for r, (sg, tag) in enumerate([(seg, 'BEFORE'), (seg_w, 'AFTER')]):
        ax[r][c].imshow(bg, cmap='gray', origin='lower', aspect='equal')
        for li, ri, nm, col in KEY:
            am = np.isin(take(atlas_seg, c, i), [li, ri]).astype(float)
            bm = np.isin(take(sg, c, i), [li, ri]).astype(float)
            if am.any():
                ax[r][c].contour(am, levels=[0.5], colors=[col], linewidths=1.5)
            if bm.any():
                ax[r][c].contour(bm, levels=[0.5], colors=[col], linewidths=1.0,
                                 linestyles='dashed')
        ax[r][c].axis('off')
        ax[r][c].set_aspect('equal')
        if r == 0:
            ax[r][c].set_title(t, fontsize=11)
        if c == 0:
            ax[r][c].text(-0.06, 0.5, tag, transform=ax[r][c].transAxes,
                          rotation=90, va='center', ha='center',
                          fontsize=12, fontweight='bold')
from matplotlib.lines import Line2D

# Windows 主控台預設 cp950，印到 emoji 會 UnicodeEncodeError 直接中斷程式。
# 不要求使用者記得設 PYTHONIOENCODING —— 忘一次就白跑一輪。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
handles = [Line2D([0], [0], color=c, lw=2, label=n) for _, _, n, c in KEY]
handles += [Line2D([0], [0], color='k', lw=2, label='atlas (solid)'),
            Line2D([0], [0], color='k', lw=1.2, ls='dashed', label='subject (dashed)')]
fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('%s — structure outlines,  solid = atlas,  dashed = subject' % name,
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.04, 1, 1])
p2 = os.path.join(out_dir, 'contours_%s_%s.png' % (name, epoch))
plt.savefig(p2, dpi=130, bbox_inches='tight')
plt.close()
print('[OK] %s' % p2)

# ── 圖 3：全部 30 個結構的 Dice 對照 ─────────────────────────────────
# 輪廓圖一次只能看到切面上有的結構；這張把 30 個全部列出來，
# 補上空間圖看不到的部分。
per = []
for l in LABELS:
    def _d(x):
        A, B = (x == l), (atlas_seg == l)
        t = A.sum() + B.sum()
        return 2.0 * (A & B).sum() / t if t else np.nan
    per.append((l, _d(seg), _d(seg_w)))
per.sort(key=lambda x: x[2] - x[1])          # 依改善幅度排序

fig, axb = plt.subplots(figsize=(11, 9))
y = np.arange(len(per))
b = [x[1] for x in per]
a2 = [x[2] for x in per]
axb.barh(y - 0.2, b, height=0.38, color='#adb5bd', label='before (affine only)')
axb.barh(y + 0.2, a2, height=0.38, color='#1f77b4', label='after (+ VoxelMorph)')
for k, (l, bb, aa) in enumerate(per):
    axb.text(aa + 0.008, k + 0.2, '%+.3f' % (aa - bb), va='center', fontsize=7.5,
             color='#1f77b4' if aa >= bb else '#d62728')
axb.set_yticks(y)
axb.set_yticklabels(['%s (%d)' % (FS_NAME.get(l, '?'), l) for l, _, _ in per], fontsize=8.5)
axb.set_xlabel('Dice')
axb.set_xlim(0, 1.0)
axb.axvline(np.nanmean(b), color='#adb5bd', ls='--', lw=1,
            label='mean before %.3f' % np.nanmean(b))
axb.axvline(np.nanmean(a2), color='#1f77b4', ls='--', lw=1,
            label='mean after %.3f' % np.nanmean(a2))
axb.legend(fontsize=9, loc='lower right')
axb.grid(axis='x', alpha=0.3)
axb.set_title('%s — all %d evaluated structures (sorted by improvement)'
              % (name, len(per)), fontsize=12, fontweight='bold')
plt.tight_layout()
p3 = os.path.join(out_dir, 'perstruct_%s_%s.png' % (name, epoch))
plt.savefig(p3, dpi=130, bbox_inches='tight')
plt.close()
print('[OK] %s' % p3)

# ── 逐結構改善 ───────────────────────────────────────────────────────
print()
print('  %-6s %-20s %7s %7s %8s' % ('label', 'name', '前', '後', '改善'))
imp = []
for l in LABELS:
    def dv(x):
        A, B = (x == l), (atlas_seg == l)
        t = A.sum() + B.sum()
        return 2.0 * (A & B).sum() / t if t else np.nan
    b, af = dv(seg), dv(seg_w)
    imp.append((l, b, af, af - b))
for l, b, af, i in sorted(imp, key=lambda x: -x[3])[:5]:
    print('  %-6d %-20s %7.4f %7.4f %+8.4f' % (l, FS_NAME.get(l,'?'), b, af, i))
for l, b, af, i in sorted(imp, key=lambda x: x[3])[:3]:
    print('  %-6d %-20s %7.4f %7.4f %+8.4f' % (l, FS_NAME.get(l,'?'), b, af, i))
