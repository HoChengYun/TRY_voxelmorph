"""
IXI 測試腳本（無 seg 版）
計算 NCC 和 SSIM，不需要 segmentation label

用法：
    python eval_ixi.py \
        --model   models/exp2_IXI/0100.pt \
        --atlas   IXI/atlas_mni152_09c_resize.npz \
        --test-dir IXI/IXI_preprocessed/test \
        --gpu     0
"""

import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND']     = 'pytorch'

import argparse
import numpy as np
import torch
import voxelmorph as vxm
from skimage.metrics import structural_similarity as ssim

# ── 參數 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model',    required=True,  help='模型路徑，例如 models/exp2_IXI/0100.pt')
parser.add_argument('--atlas',    required=True,  help='atlas npz 路徑')
parser.add_argument('--test-dir', required=True,  help='test 資料夾路徑')
parser.add_argument('--gpu',      default='0',    help='GPU ID，-1 表示 CPU')
parser.add_argument('--out-csv',  default=None,   help='輸出 CSV 路徑（預設：模型同目錄）')
args = parser.parse_args()

# ── 裝置 ──────────────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
print(f'裝置：{device}')

# ── 載入 atlas ────────────────────────────────────────────────────────
atlas_data = np.load(args.atlas)
atlas_vol  = atlas_data['vol'].astype(np.float32)
inshape    = atlas_vol.shape
print(f'Atlas shape：{inshape}')

atlas_tensor = torch.from_numpy(atlas_vol)[None, None].to(device)  # (1,1,D,H,W)

# ── 載入模型 ──────────────────────────────────────────────────────────
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()
print(f'模型載入：{args.model}')

# ── 取得測試檔案清單 ───────────────────────────────────────────────────
test_files = sorted([
    os.path.join(args.test_dir, f)
    for f in os.listdir(args.test_dir)
    if f.endswith('.npz')
])
print(f'測試筆數：{len(test_files)}')
print('=' * 60)

# ── NCC 函數 ──────────────────────────────────────────────────────────
def ncc(a, b):
    """Normalized Cross-Correlation，範圍 [-1, 1]，越接近 -1（或 1）越相似"""
    a = a.flatten()
    b = b.flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum()) + 1e-8
    return float(np.dot(a, b) / denom)

# ── 逐張評估 ──────────────────────────────────────────────────────────
results = []
atlas_np = atlas_vol  # (D,H,W)

with torch.no_grad():
    for i, fpath in enumerate(test_files):
        fname = os.path.basename(fpath)

        # 載入影像
        vol = np.load(fpath)['vol'].astype(np.float32)
        vol_tensor = torch.from_numpy(vol)[None, None].to(device)  # (1,1,D,H,W)

        # 推論
        moved, flow = model(vol_tensor, atlas_tensor, registration=True)

        # 轉回 numpy
        moved_np = moved[0, 0].cpu().numpy()  # (D,H,W)

        # 計算 NCC
        ncc_val = ncc(moved_np, atlas_np)

        # 計算 SSIM（skimage 預設 data_range 用實際範圍）
        data_range = max(moved_np.max(), atlas_np.max()) - min(moved_np.min(), atlas_np.min())
        ssim_val = ssim(moved_np, atlas_np, data_range=data_range)

        results.append({'file': fname, 'ncc': ncc_val, 'ssim': ssim_val})

        print(f'[{i+1:3d}/{len(test_files)}]  {fname:<35s}  NCC={ncc_val:+.4f}  SSIM={ssim_val:.4f}')

# ── 統計 ──────────────────────────────────────────────────────────────
ncc_vals  = np.array([r['ncc']  for r in results])
ssim_vals = np.array([r['ssim'] for r in results])

print('=' * 60)
print(f'NCC   mean ± std：{ncc_vals.mean():+.4f} ± {ncc_vals.std():.4f}')
print(f'SSIM  mean ± std：{ssim_vals.mean():.4f} ± {ssim_vals.std():.4f}')
print('=' * 60)

# ── 存 CSV ────────────────────────────────────────────────────────────
if args.out_csv is None:
    model_dir  = os.path.dirname(args.model)
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    args.out_csv = os.path.join(model_dir, f'eval_{model_name}.csv')

import csv
with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'ncc', 'ssim'])
    writer.writeheader()
    writer.writerows(results)
    writer.writerow({'file': 'MEAN', 'ncc': ncc_vals.mean(), 'ssim': ssim_vals.mean()})
    writer.writerow({'file': 'STD',  'ncc': ncc_vals.std(),  'ssim': ssim_vals.std()})

print(f'CSV 已存：{args.out_csv}')
