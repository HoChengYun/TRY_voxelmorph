"""
將 MNI152 2009c .nii 製作成 atlas_mni152_09c.npz

用法：
    python IXI\make_atlas.py --t1 你的路徑\mni_icbm152_t1_tal_nlin_asym_09c.nii
                             --mask 你的路徑\mni_icbm152_t1_tal_nlin_asym_09c_mask.nii
                             --target-shape 192,224,192   # 可選，預設輸出原始大小
                             --save-nii                   # 可選，額外輸出 .nii.gz 供驗證
"""

import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--t1',           required=True, help='T1 .nii 路徑')
parser.add_argument('--mask',         required=True, help='brain mask .nii 路徑')
parser.add_argument('--out',          default=os.path.join(_HERE, 'atlas_mni152_09c.npz'),
                    help='輸出 npz 路徑')
parser.add_argument('--target-shape', default=None,
                    help='resize 目標大小，格式 192,224,192（必須能被 16 整除）')
parser.add_argument('--save-nii', action='store_true', default=False,
                    help='額外輸出 .nii.gz 檔案，用來驗證方向和 spacing')
args = parser.parse_args()

import ants
from scipy.ndimage import zoom

print(f'載入 T1  ：{args.t1}')
print(f'載入 mask：{args.mask}')

t1   = ants.image_read(args.t1)
mask = ants.image_read(args.mask)

print(f'原始 shape  ：{t1.shape}')
print(f'原始 spacing：{t1.spacing}')

# 套 mask → 去顱骨
brain = ants.mask_image(t1, mask)

# 轉 numpy，正規化到 [0, 1]
arr = brain.numpy().astype(np.float32)
p1, p99 = np.percentile(arr[arr > 0], [1, 99])
arr = np.clip(arr, p1, p99)
arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
arr = arr.astype(np.float32)

print(f'去顱骨後 shape：{arr.shape}')

# resize（若有指定 --target-shape）
# 注意：這裡用 scipy.ndimage.zoom（雙線性插值縮放）
#   - 不是補零（zero-padding）
#   - 不是裁切（cropping）
#   - 是對每個 voxel 重新計算插值後的值
#   - 不會改變軸順序或翻轉，方向不會跑掉
if args.target_shape is not None:
    target = tuple(int(x) for x in args.target_shape.split(','))
    factors = tuple(t / s for t, s in zip(target, arr.shape))
    print(f'resize：{arr.shape} → {target}')
    print(f'  zoom factors = {tuple(round(f, 4) for f in factors)}')
    print(f'  方法 = scipy.ndimage.zoom (bilinear interpolation, order=1)')
    arr = zoom(arr, factors, order=1).astype(np.float32)

print(f'處理後 shape：{arr.shape}')
print(f'min/max     ：{arr.min():.4f} / {arr.max():.4f}')

# 存成 npz
np.savez_compressed(args.out, vol=arr)
print(f'\n✓ 儲存 npz：{args.out}')

# 額外存 .nii.gz（用來驗證方向）
if args.save_nii:
    import nibabel as nib
    nii_path = args.out.replace('.npz', '.nii.gz')
    # 用 identity affine（spacing=1mm，和 preprocess_ixi.py 裡 ants.from_numpy 一致）
    nii_img = nib.Nifti1Image(arr, np.diag([1.0, 1.0, 1.0, 1.0]))
    nib.save(nii_img, nii_path)
    print(f'✓ 儲存 nii：{nii_path}')
    print(f'  → 可以用 ITK-SNAP 或 3D Slicer 開啟驗證方向')

print()
print('下一步，執行前處理：')
print(f'  python IXI\\preprocess_ixi.py --atlas {args.out}')
