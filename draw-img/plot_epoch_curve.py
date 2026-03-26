"""
逐 epoch 評估腳本
跑 models/ 資料夾裡所有 .pt，畫出 NCC / SSIM vs Epoch 曲線

用法：
    python draw-img/plot_epoch_curve.py \
        --model-dir models/exp2_IXI \
        --atlas     IXI/atlas_mni152_09c_resize.npz \
        --test-dir  IXI/IXI_preprocessed/test \
        --out-dir   draw-img/output \
        --gpu       0

    # 只跑部分 epoch（例如每 10 個）
    python draw-img/plot_epoch_curve.py ... --step 10
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
import csv
from skimage.metrics import structural_similarity as ssim_fn

# ── 參數 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', required=True,  help='存放 .pt 的資料夾')
parser.add_argument('--atlas',     required=True)
parser.add_argument('--test-dir',  required=True)
parser.add_argument('--out-dir',   required=True)
parser.add_argument('--step',      type=int, default=1,  help='每幾個 epoch 評估一次（預設全跑）')
parser.add_argument('--gpu',       default='0')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
print(f'裝置：{device}')

# ── 載入 atlas ────────────────────────────────────────────────────────
atlas_vol    = np.load(args.atlas)['vol'].astype(np.float32)
atlas_tensor = torch.from_numpy(atlas_vol)[None, None].to(device)

# ── 測試檔案清單 ──────────────────────────────────────────────────────
test_files = sorted([
    os.path.join(args.test_dir, f)
    for f in os.listdir(args.test_dir) if f.endswith('.npz')
])
print(f'測試筆數：{len(test_files)}')

# ── 找出所有 .pt 並排序 ───────────────────────────────────────────────
pt_files = sorted([
    f for f in os.listdir(args.model_dir) if f.endswith('.pt')
])
# 每 step 個取一個
pt_files = pt_files[::args.step]
print(f'評估 epoch 數：{len(pt_files)}')

# ── NCC 函數 ──────────────────────────────────────────────────────────
def ncc(a, b):
    a, b = a.flatten(), b.flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-8))

# ── 逐 epoch 評估 ────────────────────────────────────────────────────
epochs, ncc_means, ssim_means = [], [], []
all_rows = []

for pt_name in pt_files:
    epoch_str = os.path.splitext(pt_name)[0]
    try:
        epoch_num = int(epoch_str)
    except ValueError:
        epoch_num = epoch_str

    model_path = os.path.join(args.model_dir, pt_name)
    print(f'\n── epoch {epoch_num}  ({pt_name}) ──')

    model = vxm.networks.VxmDense.load(model_path, device)
    model.to(device)
    model.eval()

    ncc_vals, ssim_vals = [], []

    with torch.no_grad():
        for fpath in test_files:
            vol = np.load(fpath)['vol'].astype(np.float32)
            vol_tensor = torch.from_numpy(vol)[None, None].to(device)
            moved, _ = model(vol_tensor, atlas_tensor, registration=True)
            moved_np = moved[0, 0].cpu().numpy()

            ncc_vals.append(ncc(moved_np, atlas_vol))
            dr = max(moved_np.max(), atlas_vol.max()) - min(moved_np.min(), atlas_vol.min())
            ssim_vals.append(ssim_fn(moved_np, atlas_vol, data_range=dr))

    ncc_mean  = np.mean(ncc_vals)
    ssim_mean = np.mean(ssim_vals)
    print(f'  NCC={ncc_mean:.4f}  SSIM={ssim_mean:.4f}')

    epochs.append(epoch_num)
    ncc_means.append(ncc_mean)
    ssim_means.append(ssim_mean)
    all_rows.append({'epoch': epoch_num, 'ncc': ncc_mean, 'ssim': ssim_mean})

# ── 存 CSV ────────────────────────────────────────────────────────────
csv_path = os.path.join(args.out_dir, 'epoch_curve.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch', 'ncc', 'ssim'])
    writer.writeheader()
    writer.writerows(all_rows)
print(f'\nCSV 已存：{csv_path}')

# ── 畫曲線圖 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_facecolor('#0D1117')

for ax in axes:
    ax.set_facecolor('#161B22')
    ax.tick_params(colors='#8B949E')
    ax.xaxis.label.set_color('#8B949E')
    ax.yaxis.label.set_color('#8B949E')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363D')

# NCC
axes[0].plot(epochs, ncc_means, color='#58A6FF', linewidth=2, marker='o', markersize=4)
axes[0].set_title('NCC vs Epoch', color='white', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('NCC')
axes[0].grid(True, color='#30363D', linestyle='--', alpha=0.5)

# SSIM
axes[1].plot(epochs, ssim_means, color='#3FB950', linewidth=2, marker='o', markersize=4)
axes[1].set_title('SSIM vs Epoch', color='white', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('SSIM')
axes[1].grid(True, color='#30363D', linestyle='--', alpha=0.5)

plt.tight_layout()
out_path = os.path.join(args.out_dir, 'epoch_curve.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'圖片已存：{out_path}')
