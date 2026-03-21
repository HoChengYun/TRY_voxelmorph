"""
VoxelMorph PyTorch 快速測試腳本
================================
在已安裝好環境後，直接用 voxelmorph-code/data/ 裡的範例資料測試。

使用方式（在 claude_cheng 資料夾內執行）：
    python voxelmorph_quick_test.py

需要先安裝：
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install scipy scikit-image nibabel h5py neurite
    pip install -e voxelmorph-code/
"""

import os
import sys
import time
import numpy as np
import torch

# ---- 設定 PyTorch 後端 ----
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

# ---- 路徑設定 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'voxelmorph-code', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'voxelmorph_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print("  VoxelMorph PyTorch 快速測試")
print("=" * 50)
print(f"PyTorch 版本 : {torch.__version__}")
print(f"CUDA 可用    : {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用裝置     : {device}")
print()

# ============================================================
# 1. 載入範例資料
# ============================================================
print("[步驟 1] 載入範例資料")

moving_path = os.path.join(DATA_DIR, 'test_scan.npz')
atlas_path  = os.path.join(DATA_DIR, 'atlas.npz')
labels_path = os.path.join(DATA_DIR, 'labels.npz')

# 檢查檔案是否存在
for path in [moving_path, atlas_path]:
    if not os.path.exists(path):
        print(f"  ✗ 找不到檔案：{path}")
        sys.exit(1)

# 載入影像
moving = vxm.py.utils.load_volfile(moving_path, np_var='vol',
                                    add_batch_axis=True, add_feat_axis=True)
fixed  = vxm.py.utils.load_volfile(atlas_path,  np_var='vol',
                                    add_batch_axis=True, add_feat_axis=True)

# 嘗試載入分割標籤（用於評估，不一定存在）
try:
    moving_seg = np.load(moving_path)['seg']
    atlas_seg  = np.load(atlas_path)['seg']
    labels     = np.load(labels_path)['labels']
    has_seg = True
    print(f"  ✓ 影像 shape    : {moving.shape}")
    print(f"  ✓ 分割標籤已載入 (labels: {labels})")
except KeyError:
    has_seg = False
    print(f"  ✓ 影像 shape    : {moving.shape}")
    print(f"  ℹ 無分割標籤（略過 Dice 評估）")

inshape = moving.shape[1:-1]
print(f"  ✓ 影像空間維度  : {inshape}")
print()

# ============================================================
# 2. 建立模型
# ============================================================
print("[步驟 2] 建立 VxmDense 模型")

model = vxm.networks.VxmDense(
    inshape=inshape,
    nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
    int_steps=7,
    int_downsize=2,
    bidir=False
)
model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ 模型建立完成")
print(f"  ✓ 總參數量 : {total_params:,}")
print()

# ============================================================
# 3. 資料轉換成 tensor
# ============================================================
input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
input_fixed  = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

# ============================================================
# 4. 執行配準推論
# ============================================================
print("[步驟 3] 執行配準推論（隨機初始化權重）")

t0 = time.time()
with torch.no_grad():
    moved, warp = model(input_moving, input_fixed, registration=True)
elapsed = time.time() - t0

moved_np = moved.detach().cpu().numpy().squeeze()
warp_np  = warp.detach().cpu().numpy().squeeze()

print(f"  ✓ 推論完成")
print(f"  ✓ 推論時間         : {elapsed:.3f} 秒")
print(f"  ✓ Warped 影像 shape : {moved_np.shape}")
print(f"  ✓ Warp field shape  : {warp_np.shape}")
print(f"  ✓ Warped 強度範圍   : [{moved_np.min():.4f}, {moved_np.max():.4f}]")
print(f"  ✓ Warp 位移範圍     : [{warp_np.min():.4f}, {warp_np.max():.4f}]")
print()

# ============================================================
# 5. 計算 Jacobian 行列式（衡量形變品質）
# ============================================================
print("[步驟 4] 計算 Jacobian 行列式")

# warp_np 的 shape 是 (3, D, H, W)，需要轉成 (D, H, W, 3)
jac_input = warp_np.transpose(1, 2, 3, 0)
jac_det = vxm.py.utils.jacobian_determinant(jac_input)
neg_frac = np.mean(jac_det < 0) * 100

print(f"  ✓ Jacobian mean    : {np.mean(jac_det):.4f}")
print(f"  ✓ Jacobian min/max : {np.min(jac_det):.4f} / {np.max(jac_det):.4f}")
print(f"  ✓ 負 Jacobian 比例  : {neg_frac:.2f}%")
if neg_frac > 10:
    print(f"  ⚠ 負比例偏高屬正常（未訓練模型），訓練後應 < 1%")
print()

# ============================================================
# 6. 計算 Dice（如果有分割標籤）
# ============================================================
if has_seg:
    print("[步驟 5] 計算 Dice 係數")

    # 用 nearest-neighbor 插值 warp 分割標籤
    seg_tensor = torch.from_numpy(
        moving_seg[np.newaxis, np.newaxis, ...].astype(np.float32)
    ).to(device)
    transformer = vxm.torch.layers.SpatialTransformer(inshape, mode='nearest')
    transformer.to(device)

    with torch.no_grad():
        warped_seg = transformer(seg_tensor, warp).squeeze().cpu().numpy()

    # 計算 Dice
    dice_scores = vxm.py.utils.dice(warped_seg.astype(int),
                                      atlas_seg.astype(int), labels=labels)
    print(f"  ✓ 平均 Dice : {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"  ⚠ 此分數為隨機模型，訓練後通常可達 0.7~0.8 以上")
    print()

# ============================================================
# 7. 儲存結果
# ============================================================
print("[步驟 6] 儲存輸出結果")

warped_path = os.path.join(OUTPUT_DIR, 'warped_image.npz')
warp_path   = os.path.join(OUTPUT_DIR, 'warp_field.npz')

np.savez_compressed(warped_path, vol=moved_np)
np.savez_compressed(warp_path,   vol=warp_np)

print(f"  ✓ Warped 影像 → {warped_path}")
print(f"  ✓ Warp field  → {warp_path}")
print()

# ============================================================
# 完成
# ============================================================
print("=" * 50)
print("  測試完成！VoxelMorph 環境正常運作。")
print()
print("  下一步：")
print("  1. 準備訓練資料（腦部 MRI .npz）")
print("  2. 執行訓練腳本：")
print("     python voxelmorph-code/scripts/torch/train.py \\")
print("         oasis/oasis_npz/train --atlas voxelmorph-code/data/atlas.npz \\")
print("         --model-dir models/ --image-loss ncc --lambda 1.0")
print("=" * 50)
