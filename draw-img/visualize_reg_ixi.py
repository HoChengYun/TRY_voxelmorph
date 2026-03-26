"""
配準視覺化腳本
輸出論文常見的四格圖：Source / Atlas / Warped / Difference

用法：
    python draw-img/visualize_reg.py \
        --model    models/exp2_IXI/0100.pt \
        --atlas    IXI/atlas_mni152_09c_resize.npz \
        --subject  IXI/IXI_preprocessed/test/IXI017-Guys-0698-T1.npz \
        --out-dir  draw-img/output \
        --gpu      0

    # 不指定 --subject 則隨機從 test-dir 選一張
    python draw-img/visualize_reg.py \
        --model    models/exp2_IXI/0100.pt \
        --atlas    IXI/atlas_mni152_09c_resize.npz \
        --test-dir IXI/IXI_preprocessed/test \
        --out-dir  draw-img/output \
        --gpu      0
"""

import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND']     = 'pytorch'

import argparse
import numpy as np
import torch
import voxelmorph as vxm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 參數 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model',    required=True)
parser.add_argument('--atlas',    required=True)
parser.add_argument('--subject',  default=None,  help='指定單張 npz，不指定則從 test-dir 隨機選')
parser.add_argument('--test-dir', default=None)
parser.add_argument('--slice-axis', default='axial', choices=['axial','coronal','sagittal'],
                    help='切面方向（預設 axial）')
parser.add_argument('--out-dir',  required=True)
parser.add_argument('--gpu',      default='0')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')

# ── 載入 atlas ────────────────────────────────────────────────────────
atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
atlas_tensor = torch.from_numpy(atlas_vol)[None, None].to(device)

# ── 載入模型 ──────────────────────────────────────────────────────────
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()

# ── 選擇受試者 ────────────────────────────────────────────────────────
if args.subject is not None:
    subject_path = args.subject
else:
    import random
    files = [os.path.join(args.test_dir, f)
             for f in os.listdir(args.test_dir) if f.endswith('.npz')]
    subject_path = random.choice(files)

subject_name = os.path.splitext(os.path.basename(subject_path))[0]
print(f'受試者：{subject_name}')

vol = np.load(subject_path)['vol'].astype(np.float32)
vol_tensor = torch.from_numpy(vol)[None, None].to(device)

# ── 推論 ──────────────────────────────────────────────────────────────
with torch.no_grad():
    moved, flow = model(vol_tensor, atlas_tensor, registration=True)

moved_np = moved[0, 0].cpu().numpy()
flow_np  = flow[0].cpu().numpy()   # (3, D, H, W)

# ── 取中間切面 ────────────────────────────────────────────────────────
D, H, W = vol.shape

def get_slice(vol3d, axis, idx):
    if axis == 'axial':    return vol3d[idx, :, :]
    if axis == 'coronal':  return vol3d[:, idx, :]
    if axis == 'sagittal': return vol3d[:, :, idx]

mid = {'axial': D//2, 'coronal': H//2, 'sagittal': W//2}[args.slice_axis]

src_sl   = get_slice(vol,       args.slice_axis, mid)
atl_sl   = get_slice(atlas_vol, args.slice_axis, mid)
warp_sl  = get_slice(moved_np,  args.slice_axis, mid)
diff_sl  = np.abs(warp_sl - atl_sl)

# Atlas / Warped 對齊 Source 的左右方向
# imshow 用 .T 轉置，所以 flipud（翻行）才會在顯示上變成左右翻轉
atl_sl  = np.flipud(atl_sl)
warp_sl = np.flipud(warp_sl)
diff_sl = np.flipud(diff_sl)

# ── 計算指標 ──────────────────────────────────────────────────────────
from skimage.metrics import structural_similarity as ssim_fn

def ncc(a, b):
    a, b = a.flatten(), b.flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-8))

ncc_val  = ncc(moved_np, atlas_vol)
dr       = max(moved_np.max(), atlas_vol.max()) - min(moved_np.min(), atlas_vol.min())
ssim_val = ssim_fn(moved_np, atlas_vol, data_range=dr)
mad_val  = float(np.mean(diff_sl))          # mean absolute difference（切面）
diff_max = float(diff_sl.max())

# ── 繪圖 ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5.2))
fig.patch.set_facecolor('#0D1117')

# 前三格等寬，第四格留空間給 colorbar
gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.06,
                       left=0.02, right=0.96, top=0.84, bottom=0.05,
                       width_ratios=[1, 1, 1, 1.12])

titles = ['Source', 'Atlas', 'Warped', 'Difference  |Warped − Atlas|']
slices = [src_sl, atl_sl, warp_sl, diff_sl]
cmaps  = ['gray', 'gray', 'gray', 'hot']

axes = []
for i, (title, sl, cmap) in enumerate(zip(titles, slices, cmaps)):
    ax = fig.add_subplot(gs[i])
    im = ax.imshow(sl.T, cmap=cmap, origin='lower', aspect='equal',
                   vmin=0, vmax=1 if cmap == 'gray' else diff_sl.max())
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=6)
    ax.axis('off')

    # Difference 圖加 colorbar + 統計數字
    if i == 3:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='#8B949E', labelcolor='#8B949E')
        cbar.outline.set_edgecolor('#30363D')
        ax.text(0.05, 0.04,
                f'MAD = {mad_val:.4f}\nmax = {diff_max:.4f}',
                transform=ax.transAxes,
                color='white', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#161B22', alpha=0.8))
    axes.append(ax)

model_name = os.path.splitext(os.path.basename(args.model))[0]
fig.suptitle(
    f'{subject_name}   |   model: {model_name}   |   '
    f'NCC={ncc_val:.4f}   SSIM={ssim_val:.4f}   [{args.slice_axis}  slice={mid}]',
    color='white', fontsize=11, y=0.96
)

out_path = os.path.join(args.out_dir, f'reg_{subject_name}_{model_name}_{args.slice_axis}.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'✓ 儲存：{out_path}')
