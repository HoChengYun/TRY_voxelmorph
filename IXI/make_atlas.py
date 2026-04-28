"""
將 MNI152 2009c .nii 製作成 atlas npz + nii.gz

用法：
    python IXI\make_atlas.py --t1 你的路徑\mni_icbm152_t1_tal_nlin_asym_09c.nii
                             --mask 你的路徑\mni_icbm152_t1_tal_nlin_asym_09c_mask.nii
                             --target-shape 192,224,192

輸出（兩個都會產生）：
    atlas_mni152_09c.npz       ← 給 VoxelMorph train.py 用（只有 numpy array）
    atlas_mni152_09c.nii.gz    ← 給 preprocess_ixi.py 用（帶完整 header）
"""

import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--t1',           required=True, help='T1 .nii 路徑')
parser.add_argument('--mask',         required=True, help='brain mask .nii 路徑')
parser.add_argument('--out',          default=os.path.join(_HERE, 'atlas_mni152_09c'),
                    help='輸出路徑（不含副檔名，會同時產生 .npz 和 .nii.gz）')
parser.add_argument('--target-shape', default=None,
                    help='resize 目標大小，格式 192,224,192（必須能被 16 整除）')
args = parser.parse_args()

import ants

print(f'載入 T1  ：{args.t1}')
print(f'載入 mask：{args.mask}')

t1   = ants.image_read(args.t1)
mask = ants.image_read(args.mask)

print(f'原始 shape  ：{t1.shape}')
print(f'原始 spacing：{t1.spacing}')
print(f'原始 origin ：{t1.origin}')
print(f'原始 direction：')
print(f'{t1.direction}')

# 套 mask → 去顱骨
brain = ants.mask_image(t1, mask)

# 轉 numpy，正規化到 [0, 1]
arr = brain.numpy().astype(np.float32)
p1, p99 = np.percentile(arr[arr > 0], [1, 99])
arr = np.clip(arr, p1, p99)
arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
arr = arr.astype(np.float32)

# 把正規化後的數值放回 ANTs image（保留原始 header）
brain_norm = brain.new_image_like(arr)

print(f'去顱骨+正規化後 shape：{brain_norm.shape}')

# resize（若有指定 --target-shape）
# 使用 ANTs resample_image，保留 header（origin, direction, 調整 spacing）
# 這比 scipy.ndimage.zoom 好，因為不會丟失空間資訊
if args.target_shape is not None:
    target = tuple(int(x) for x in args.target_shape.split(','))
    print(f'resize：{brain_norm.shape} → {target}')
    print(f'  方法 = ants.resample_image (保留 header)')
    
    brain_resized = ants.resample_image(
        brain_norm,
        target,            # 目標 voxel 數量
        use_voxels=True,   # True = 參數是 voxel 數量，不是 spacing
        interp_type=1,     # 1 = linear interpolation
    )
else:
    brain_resized = brain_norm

final_arr = brain_resized.numpy().astype(np.float32)

print(f'處理後 shape  ：{brain_resized.shape}')
print(f'處理後 spacing：{brain_resized.spacing}')
print(f'處理後 origin ：{brain_resized.origin}')
print(f'處理後 direction：')
print(f'{brain_resized.direction}')
print(f'min/max       ：{final_arr.min():.4f} / {final_arr.max():.4f}')

# 移除副檔名（如果使用者帶了 .npz 或 .nii.gz）
out_base = args.out
for ext in ['.npz', '.nii.gz', '.nii']:
    if out_base.endswith(ext):
        out_base = out_base[:-len(ext)]
        break

# 1. 存 .nii.gz（帶完整 MNI152 header，給 preprocess_ixi.py 的 ANTs 用）
nii_path = out_base + '.nii.gz'
ants.image_write(brain_resized, nii_path)
print(f'\n✓ 儲存 nii.gz：{nii_path}  （帶 header，給 preprocess_ixi.py --atlas 用）')

# 2. 存 .npz（只有 numpy array，給 VoxelMorph train.py 用）
npz_path = out_base + '.npz'
np.savez_compressed(npz_path, vol=final_arr)
print(f'✓ 儲存 npz   ：{npz_path}  （給 train.py --atlas 用）')

print()
print('下一步：')
print(f'  # 前處理 IXI（用帶 header 的 .nii.gz atlas）')
print(f'  python IXI\\preprocess_ixi.py --atlas {nii_path}')
print()
print(f'  # 訓練 VoxelMorph（用 .npz atlas）')
print(f'  python voxelmorph-code\\scripts\\torch\\train.py --atlas {npz_path} ...')
