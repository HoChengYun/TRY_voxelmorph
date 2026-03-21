"""
VoxelMorph 配準結果視覺化
==========================
產生四張圖，存到 draw-img/{model名稱}/ 資料夾：

  overview  : Moving / Atlas / Warped / |差值前| / |差值後|（直覺看配準效果）
  fig4      : Moving / Fixed / Warped + 分割邊界疊圖（論文 Fig.4）
  fig5      : per-structure Dice boxplot（論文 Fig.5，需提供 --csv）
  fig6      : 彩色位移場 + 格線形變圖（論文 Fig.6）

使用方式（從 claude_cheng 根目錄執行）：
    python draw-img/visualize_registration.py --model models/0004.pt
    python draw-img/visualize_registration.py --model models/0004.pt --csv models/dice_0004.csv
"""

import os
import glob
import random
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

# ============================================================
# 路徑()
# ============================================================
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.normpath(os.path.join(_HERE, '..'))

DEFAULT_ATLAS    = os.path.normpath(os.path.join(_ROOT, 'voxelmorph-code', 'data', 'atlas.npz'))
DEFAULT_TEST_DIR = os.path.normpath(os.path.join(_ROOT, 'oasis', 'oasis_npz', 'test'))

# seg35 → FreeSurfer ID
SEG35_TO_FS = {
     0:  0,   1:  2,   2:  3,   3:  4,   4:  5,
     5:  7,   6:  8,   7: 10,   8: 11,   9: 12,
    10: 13,  11: 14,  12: 15,  13: 16,  14: 17,
    15: 18,  16: 26,  17: 28,  18: 30,  19: 31,
    20: 41,  21: 42,  22: 43,  23: 44,  24: 46,
    25: 47,  26: 49,  27: 50,  28: 51,  29: 52,
    30: 53,  31: 54,  32: 58,  33: 60,  34: 62,
    35: 63,
}

# 論文 Fig.4 疊圖結構（藍=腦室、紅=丘腦、綠=海馬）
OVERLAY_STRUCTS = {
     4: ('#3399FF', 'Lat-Ventricle-L'),
    43: ('#3399FF', 'Lat-Ventricle-R'),
    10: ('#FF6666', 'Thalamus-L'),
    49: ('#FF6666', 'Thalamus-R'),
    17: ('#66CC66', 'Hippocampus-L'),
    53: ('#66CC66', 'Hippocampus-R'),
}

def remap_seg(seg):
    out = np.zeros_like(seg)
    for src, dst in SEG35_TO_FS.items():
        out[seg == src] = dst
    return out

def seg_to_boundary(seg_slice, label_id, dilation=1):
    from scipy.ndimage import binary_erosion
    mask = (seg_slice == label_id)
    if not mask.any():
        return None
    inner = binary_erosion(mask, iterations=dilation)
    return mask & ~inner

# ============================================================
# 參數
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model',    required=True)
parser.add_argument('--atlas',    default=DEFAULT_ATLAS)
parser.add_argument('--subject',  default=None, help='指定測試影像 .npz，不指定則隨機選')
parser.add_argument('--test-dir', default=DEFAULT_TEST_DIR)
parser.add_argument('--csv',      default=None, help='test.py 產生的 per-structure Dice CSV，用於畫 Fig.5')
parser.add_argument('--gpu',      default='-1')
args = parser.parse_args()

model_stem = os.path.splitext(os.path.basename(args.model))[0]

# 輸出資料夾：draw-img/{model名稱}/
out_dir = os.path.join(_HERE, model_stem)
os.makedirs(out_dir, exist_ok=True)

device = 'cuda' if args.gpu != '-1' else 'cpu'
if device == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# ============================================================
# 選擇測試影像
# ============================================================
if args.subject:
    subject_path = args.subject
else:
    test_files   = sorted(glob.glob(os.path.join(args.test_dir, '*.npz')))
    subject_path = random.choice(test_files)

subject_name = os.path.basename(subject_path).replace('.npz', '')

print(f'Subject   : {subject_name}')
print(f'Model     : {args.model}')
print(f'Device    : {device}')
print(f'Output    : {out_dir}/')
print()

# ============================================================
# 載入資料 & 推論
# ============================================================
moving_data = np.load(subject_path)
moving_vol  = moving_data['vol'].astype(np.float32)
moving_seg  = remap_seg(moving_data['seg'].astype(int))

atlas_data = np.load(args.atlas)
atlas_vol  = atlas_data['vol'].astype(np.float32)
atlas_seg  = atlas_data['seg'].astype(int)

model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()

mov_t = torch.from_numpy(moving_vol[np.newaxis, np.newaxis]).to(device).float()
fix_t = torch.from_numpy(atlas_vol[np.newaxis,  np.newaxis]).to(device).float()
seg_t = torch.from_numpy(moving_seg[np.newaxis, np.newaxis].astype(np.float32)).to(device)

transformer = vxm.torch.layers.SpatialTransformer(atlas_vol.shape, mode='nearest')
transformer.to(device)

with torch.no_grad():
    warped_t, warp_t = model(mov_t, fix_t, registration=True)
    warped_seg_t     = transformer(seg_t, warp_t)

warped_vol = warped_t.cpu().numpy().squeeze()
warped_seg = warped_seg_t.cpu().numpy().squeeze().astype(int)
warp_np    = warp_t.cpu().numpy().squeeze()  # (3, D, H, W)

mae_before = float(np.mean(np.abs(moving_vol - atlas_vol)))
mae_after  = float(np.mean(np.abs(warped_vol - atlas_vol)))
print(f'MAE before : {mae_before:.4f}')
print(f'MAE after  : {mae_after:.4f}  ({(mae_before-mae_after)/mae_before*100:.1f}% 改善)')
print()

# 切面中心
cx = moving_vol.shape[0] // 2
cy = moving_vol.shape[1] // 2
cz = moving_vol.shape[2] // 2

planes = {
    'Axial'   : {'moving': moving_vol[cx,:,:],  'fixed': atlas_vol[cx,:,:],
                 'warped': warped_vol[cx,:,:],
                 'mov_seg': moving_seg[cx,:,:],  'fix_seg': atlas_seg[cx,:,:],
                 'war_seg': warped_seg[cx,:,:],
                 'warp_x': warp_np[0,cx,:,:],   'warp_y': warp_np[1,cx,:,:]},
    'Coronal' : {'moving': moving_vol[:,cy,:],  'fixed': atlas_vol[:,cy,:],
                 'warped': warped_vol[:,cy,:],
                 'mov_seg': moving_seg[:,cy,:],  'fix_seg': atlas_seg[:,cy,:],
                 'war_seg': warped_seg[:,cy,:],
                 'warp_x': warp_np[0,:,cy,:],   'warp_y': warp_np[2,:,cy,:]},
    'Sagittal': {'moving': moving_vol[:,:,cz],  'fixed': atlas_vol[:,:,cz],
                 'warped': warped_vol[:,:,cz],
                 'mov_seg': moving_seg[:,:,cz],  'fix_seg': atlas_seg[:,:,cz],
                 'war_seg': warped_seg[:,:,cz],
                 'warp_x': warp_np[1,:,:,cz],   'warp_y': warp_np[2,:,:,cz]},
}

# ============================================================
# Overview：Moving / Atlas / Warped / |差值前| / |差值後|
# （你原本的那張圖）
# ============================================================
fig_ov, axes_ov = plt.subplots(3, 5, figsize=(18, 11))
fig_ov.suptitle(
    f'Registration Overview\n'
    f'Model: {args.model}  |  Subject: {subject_name}  |  '
    f'MAE: {mae_before:.4f} → {mae_after:.4f}  ({(mae_before-mae_after)/mae_before*100:.1f}% better)',
    fontsize=11, fontweight='bold'
)

col_titles_ov = ['Moving', 'Atlas (Fixed)', 'Warped', '|Moving − Atlas|', '|Warped − Atlas|']
for col, t in enumerate(col_titles_ov):
    axes_ov[0, col].set_title(t, fontsize=10, pad=6)

for row, (plane_name, p) in enumerate(planes.items()):
    diff_before = np.abs(p['moving'] - p['fixed'])
    diff_after  = np.abs(p['warped'] - p['fixed'])
    imgs = [p['moving'], p['fixed'], p['warped'], diff_before, diff_after]

    for col, img in enumerate(imgs):
        ax   = axes_ov[row, col]
        cmap = 'hot' if col >= 3 else 'gray'
        vmax = 0.5  if col >= 3 else 1.0
        ax.imshow(np.rot90(img), cmap=cmap, vmin=0, vmax=vmax)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(plane_name, fontsize=10)

cbar_ax = fig_ov.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=0.5))
fig_ov.colorbar(sm, cax=cbar_ax, label='Absolute Difference')
plt.tight_layout(rect=[0, 0, 0.91, 0.94])

ov_path = os.path.join(out_dir, f'overview_{subject_name}.png')
fig_ov.savefig(ov_path, dpi=120, bbox_inches='tight')
plt.close(fig_ov)
print(f'Overview → {ov_path}')

# ============================================================
# Fig.4：Moving / Fixed / Warped + 分割邊界疊圖
# ============================================================
fig4, axes4 = plt.subplots(3, 3, figsize=(13, 12))
fig4.suptitle(
    f'Fig.4 — Segmentation Boundary Overlay\n'
    f'Model: {args.model}  |  Subject: {subject_name}',
    fontsize=11, fontweight='bold'
)

for col, t in enumerate(['m  (Moving)', 'f  (Fixed / Atlas)', 'm∘φ  (Warped)']):
    axes4[0, col].set_title(t, fontsize=10, pad=6)

legend_patches = [
    Patch(color='#3399FF', label='Lateral Ventricle'),
    Patch(color='#FF6666', label='Thalamus'),
    Patch(color='#66CC66', label='Hippocampus'),
]

for row, (plane_name, p) in enumerate(planes.items()):
    for col, (img_key, seg_key) in enumerate(
            [('moving','mov_seg'), ('fixed','fix_seg'), ('warped','war_seg')]):
        ax  = axes4[row, col]
        img = np.rot90(p[img_key])
        seg = np.rot90(p[seg_key])
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        for label_id, (color, _) in OVERLAY_STRUCTS.items():
            boundary = seg_to_boundary(seg, label_id)
            if boundary is not None and boundary.any():
                rgba      = np.zeros((*boundary.shape, 4))
                rgba[boundary] = [*mcolors.to_rgb(color), 0.9]
                ax.imshow(rgba, interpolation='none')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(plane_name, fontsize=10)

fig4.legend(handles=legend_patches, loc='lower center', ncol=3,
            fontsize=9, bbox_to_anchor=(0.5, 0.01))
plt.tight_layout(rect=[0, 0.04, 1, 0.94])

fig4_path = os.path.join(out_dir, f'fig4_boundary_{subject_name}.png')
fig4.savefig(fig4_path, dpi=130, bbox_inches='tight')
plt.close(fig4)
print(f'Fig.4  → {fig4_path}')

# ============================================================
# Fig.6：彩色位移場 + 格線形變圖
# ============================================================
GRID_STEP = 8

fig6, axes6 = plt.subplots(3, 5, figsize=(20, 12))
fig6.suptitle(
    f'Fig.6 — Deformation Field\n'
    f'Model: {args.model}  |  Subject: {subject_name}',
    fontsize=11, fontweight='bold'
)

for col, t in enumerate(['m (Moving)', 'f (Fixed)', 'm∘φ (Warped)',
                          'φ (Displacement, RGB)', 'φ (Grid)']):
    axes6[0, col].set_title(t, fontsize=9, pad=6)

for row, (plane_name, p) in enumerate(planes.items()):
    for col, key in enumerate(['moving', 'fixed', 'warped']):
        ax = axes6[row, col]
        ax.imshow(np.rot90(p[key]), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(plane_name, fontsize=10)

    # 彩色位移圖
    ax = axes6[row, 3]
    dx = np.rot90(p['warp_x'])
    dy = np.rot90(p['warp_y'])
    mag = np.sqrt(dx**2 + dy**2)
    r_ch = np.clip((dx - dx.min()) / (dx.max() - dx.min() + 1e-8), 0, 1)
    g_ch = np.clip((dy - dy.min()) / (dy.max() - dy.min() + 1e-8), 0, 1)
    b_ch = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    ax.imshow(np.stack([r_ch, g_ch, b_ch], axis=-1))
    ax.axis('off')

    # 格線形變圖
    ax = axes6[row, 4]
    H, W = dx.shape
    ax.set_facecolor('black')
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    for gx in range(0, W, GRID_STEP):
        ys = np.arange(H)
        xs = gx + dy[:, gx] * 0.5
        ax.plot(xs, ys, color='#00FF88', linewidth=0.5, alpha=0.8)
    for gy in range(0, H, GRID_STEP):
        xs = np.arange(W)
        ys = gy + dx[gy, :] * 0.5
        ax.plot(xs, ys, color='#00FF88', linewidth=0.5, alpha=0.8)
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig6_path = os.path.join(out_dir, f'fig6_deformation_{subject_name}.png')
fig6.savefig(fig6_path, dpi=130, bbox_inches='tight')
plt.close(fig6)
print(f'Fig.6  → {fig6_path}')

# ============================================================
# Fig.5：per-structure Dice boxplot
# ============================================================
if args.csv and os.path.exists(args.csv):
    import csv as csv_mod

    with open(args.csv, newline='', encoding='utf-8') as f:
        rows = list(csv_mod.reader(f))

    header    = rows[0][1:]
    data_rows = [r for r in rows[1:] if r[0] not in ('MEAN', 'STD')]
    dice_mat  = np.array([[float(v) for v in r[1:]] for r in data_rows])

    # 左右半球平均（仿論文）
    def pair_avg(names, mat):
        paired, paired_names, used = [], [], set()
        for i, n in enumerate(names):
            if i in used:
                continue
            base   = n.replace('-L', '').replace('-R', '')
            j_list = [j for j, m in enumerate(names)
                      if j != i and m.replace('-L','').replace('-R','') == base]
            if j_list:
                j = j_list[0]
                paired.append((mat[:, i] + mat[:, j]) / 2)
                paired_names.append(base)
                used.add(i); used.add(j)
            elif i not in used:
                paired.append(mat[:, i])
                paired_names.append(n)
                used.add(i)
        return paired_names, paired

    paired_names, paired_data = pair_avg(header, dice_mat)
    # 字母順序排列，方便跨模型比較
    order        = np.argsort(paired_names)
    sorted_names = [paired_names[i] for i in order]
    sorted_data  = [paired_data[i]  for i in order]

    fig5, ax5 = plt.subplots(figsize=(max(12, len(sorted_names)*0.75), 5))
    bp = ax5.boxplot(sorted_data, patch_artist=True, notch=False,
                     medianprops=dict(color='black', linewidth=1.5),
                     whiskerprops=dict(linewidth=1),
                     capprops=dict(linewidth=1),
                     flierprops=dict(marker='.', markersize=3, alpha=0.5))

    colors = plt.cm.tab20.colors
    for patch, color in zip(bp['boxes'], colors * 4):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax5.set_xticks(range(1, len(sorted_names)+1))
    ax5.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('Dice Score', fontsize=11)
    ax5.set_ylim(0, 1.05)
    ax5.set_title(
        f'Fig.5 — Dice per Anatomical Structure  '
        f'(n={len(data_rows)} subjects,  overall mean={np.mean(dice_mat):.3f})\n'
        f'Model: {args.model}',
        fontsize=11, fontweight='bold'
    )
    ax5.axhline(y=np.mean(dice_mat), color='gray', linestyle='--',
                linewidth=1, label=f'Overall Mean = {np.mean(dice_mat):.3f}')
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    fig5_path = os.path.join(out_dir, f'fig5_boxplot_{model_stem}.png')
    fig5.savefig(fig5_path, dpi=130, bbox_inches='tight')
    plt.close(fig5)
    print(f'Fig.5  → {fig5_path}')
else:
    if args.csv:
        print(f'\n  找不到 CSV：{args.csv}，略過 Fig.5')
    else:
        print(f'\n  未提供 --csv，略過 Fig.5  boxplot')
        print(f'  先執行：python voxelmorph-code/scripts/torch/test.py --model {args.model}')
        print(f'  再加上：--csv models/dice_{model_stem}.csv')

# ============================================================
# 完成
# ============================================================
print()
print(f'所有圖片已存到 → {out_dir}/')
