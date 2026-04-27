"""
讀取 NIfTI (.nii / .nii.gz) 檔案的 header 資訊。

用法：
    python IXI\\read_nii_header.py  檔案1.nii.gz  檔案2.nii.gz ...
    python IXI\\read_nii_header.py  IXI\\IXI_preprocessed\\nii\\*.nii.gz
    python IXI\\read_nii_header.py  IXI\\orientation_verify\\*.nii.gz
"""

import os
import sys
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='讀取 NIfTI header 資訊')
parser.add_argument('files', nargs='+', help='.nii 或 .nii.gz 檔案路徑（可多個，支援 wildcard）')
parser.add_argument('--full', action='store_true', help='顯示完整 header（預設只顯示重點）')
args = parser.parse_args()

try:
    import nibabel as nib
except ImportError:
    print("需要 nibabel：pip install nibabel")
    sys.exit(1)

# 展開 wildcard（Windows 的 shell 不會自動展開）
all_files = []
for pattern in args.files:
    expanded = glob.glob(pattern)
    if expanded:
        all_files.extend(expanded)
    elif os.path.exists(pattern):
        all_files.append(pattern)
    else:
        print(f"找不到：{pattern}")

# 過濾只保留 .nii 和 .nii.gz
all_files = [f for f in all_files if f.endswith('.nii') or f.endswith('.nii.gz')]
all_files = sorted(set(all_files))

if not all_files:
    print("沒有找到任何 .nii / .nii.gz 檔案")
    sys.exit(1)

print(f"共 {len(all_files)} 個檔案\n")
print("=" * 90)

for i, filepath in enumerate(all_files):
    img = nib.load(filepath)
    hdr = img.header
    affine = img.affine
    
    name = os.path.basename(filepath)
    shape = img.shape
    spacing = tuple(round(float(s), 4) for s in hdr.get_zooms()[:3])
    orientation = nib.aff2axcodes(affine)
    dtype = hdr.get_data_dtype()
    
    # 取值域（不載入全部資料，只讀 header 的 cal_min / cal_max）
    cal_min = float(hdr['cal_min']) if 'cal_min' in hdr else None
    cal_max = float(hdr['cal_max']) if 'cal_max' in hdr else None
    
    print(f"[{i+1}/{len(all_files)}] {name}")
    print(f"  路徑        : {filepath}")
    print(f"  Shape       : {shape}")
    print(f"  Spacing(mm) : {spacing}")
    print(f"  Orientation : {orientation}")
    print(f"  Dtype       : {dtype}")
    print(f"  Affine      :")
    for row in affine:
        print(f"    [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}]")
    
    # 計算一些 affine 衍生資訊
    # voxel size from affine (更準確)
    voxel_sizes = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    print(f"  Voxel size  : ({voxel_sizes[0]:.4f}, {voxel_sizes[1]:.4f}, {voxel_sizes[2]:.4f}) mm")
    
    # origin
    origin = affine[:3, 3]
    print(f"  Origin      : ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")
    
    # direction (normalized columns of rotation matrix)
    direction = affine[:3, :3] / voxel_sizes[np.newaxis, :]
    is_identity = np.allclose(direction, np.eye(3), atol=0.01)
    is_diag = np.allclose(direction, np.diag(np.diag(direction)), atol=0.01)
    
    if is_identity:
        print(f"  Direction   : Identity (標準方向)")
    elif is_diag:
        diag = np.diag(direction)
        print(f"  Direction   : Diagonal {tuple(round(float(d), 2) for d in diag)}")
    else:
        print(f"  Direction   : (有旋轉分量)")
    
    if args.full:
        print(f"\n  --- 完整 Header ---")
        print(f"  {hdr}")
    
    print("-" * 90)

# 總結表
if len(all_files) > 1:
    print(f"\n{'='*90}")
    print(f"  總結（{len(all_files)} 個檔案）")
    print(f"{'='*90}")
    print(f"\n  {'檔案名稱':<45} {'Shape':<22} {'Spacing':<22} {'Orient'}")
    print(f"  {'-'*43} {'-'*20} {'-'*20} {'-'*10}")
    
    for filepath in all_files:
        img = nib.load(filepath)
        hdr = img.header
        name = os.path.basename(filepath)
        shape = img.shape
        spacing = tuple(round(float(s), 4) for s in hdr.get_zooms()[:3])
        orientation = nib.aff2axcodes(img.affine)
        print(f"  {name:<45} {str(shape):<22} {str(spacing):<22} {orientation}")
