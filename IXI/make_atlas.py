"""
將 MNI152 2009c .nii 製作成 atlas_mni152_09c.npz

用法：
    python IXI\make_atlas.py --t1 你的路徑\mni_icbm152_t1_tal_nlin_asym_09c.nii
                             --mask 你的路徑\mni_icbm152_t1_tal_nlin_asym_09c_mask.nii
"""

import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--t1',   required=True, help='T1 .nii 路徑')
parser.add_argument('--mask', required=True, help='brain mask .nii 路徑')
parser.add_argument('--out',  default=os.path.join(_HERE, 'atlas_mni152_09c.npz'),
                    help='輸出 npz 路徑')
args = parser.parse_args()

import ants

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

print(f'處理後 shape：{arr.shape}')
print(f'min/max     ：{arr.min():.4f} / {arr.max():.4f}')

np.savez_compressed(args.out, vol=arr)
print(f'\n✓ 儲存：{args.out}')
print()
print('下一步，執行前處理：')
print(f'  python IXI\\preprocess_ixi.py --atlas {args.out}')
