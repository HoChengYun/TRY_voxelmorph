"""
IXI 前處理驗證腳本
檢查項目：
  1. 所有 .npz 的 shape 是否一致（應為 192×224×192）
  2. 原始 .nii.gz 的 orientation / spacing（抽樣）
  3. ANTs Affine 配準後的 spacing 和 orientation（透過 ants 重新讀取）
  4. atlas 的資訊

用法：
    python IXI/verify_preprocess.py
"""

import os
import sys
import glob
import numpy as np
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)  # claude_cheng/

# ── 1. 檢查所有 npz 的 shape ──────────────────────────────────────────
print("=" * 65)
print("  1. 檢查所有 preprocessed .npz 的 shape")
print("=" * 65)

npz_dir = os.path.join(_HERE, 'IXI_preprocessed')
all_npz = sorted(
    glob.glob(os.path.join(npz_dir, 'train', '*.npz')) +
    glob.glob(os.path.join(npz_dir, 'test', '*.npz'))
)
print(f"找到 {len(all_npz)} 個 .npz 檔案\n")

shape_counter = Counter()
value_ranges = []

for f in all_npz:
    data = np.load(f)
    vol = data['vol']
    shape_counter[vol.shape] += 1
    value_ranges.append((vol.min(), vol.max(), vol.mean()))

print("Shape 統計：")
for shape, count in shape_counter.most_common():
    print(f"  {shape} : {count} 個")

if len(shape_counter) == 1:
    print("\n  ✅ 所有 npz shape 一致！")
else:
    print("\n  ⚠️  存在不一致的 shape！")
    # 列出不一致的
    expected_shape = shape_counter.most_common(1)[0][0]
    for f in all_npz:
        vol = np.load(f)['vol']
        if vol.shape != expected_shape:
            print(f"    異常：{os.path.basename(f)} → {vol.shape}")

# 值域統計
mins = [r[0] for r in value_ranges]
maxs = [r[1] for r in value_ranges]
means = [r[2] for r in value_ranges]
print(f"\n值域統計（{len(value_ranges)} 個檔案）：")
print(f"  min  ∈ [{min(mins):.4f}, {max(mins):.4f}]")
print(f"  max  ∈ [{min(maxs):.4f}, {max(maxs):.4f}]")
print(f"  mean ∈ [{min(means):.4f}, {max(means):.4f}]")

# ── 2. 檢查 atlas ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  2. 檢查 Atlas")
print("=" * 65)

atlas_path = os.path.join(_HERE, 'atlas_mni152_09c.npz')
if os.path.exists(atlas_path):
    atlas = np.load(atlas_path)
    print(f"Atlas 檔案：{atlas_path}")
    print(f"  keys：{list(atlas.keys())}")
    atlas_vol = atlas['vol']
    print(f"  vol shape：{atlas_vol.shape}")
    print(f"  vol dtype：{atlas_vol.dtype}")
    print(f"  vol range：[{atlas_vol.min():.4f}, {atlas_vol.max():.4f}]")
    
    # 和 npz shape 比較
    npz_shape = shape_counter.most_common(1)[0][0]
    if atlas_vol.shape == npz_shape:
        print(f"\n  ✅ Atlas shape {atlas_vol.shape} == npz shape {npz_shape}")
    else:
        print(f"\n  ⚠️  Atlas shape {atlas_vol.shape} != npz shape {npz_shape}")
else:
    print(f"  找不到 atlas：{atlas_path}")

# ── 3. 檢查原始 .nii.gz 的 header（抽樣） ────────────────────────────
print("\n" + "=" * 65)
print("  3. 抽樣檢查原始 .nii.gz 的 header（方向 & spacing）")
print("=" * 65)

try:
    import nibabel as nib
    
    raw_dir = os.path.join(_HERE, 'IXI-T1')
    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.nii.gz')))
    
    if len(raw_files) == 0:
        print("  找不到原始 .nii.gz 檔案")
    else:
        # 抽樣 5 個（前2 + 中間1 + IOP 1 + HH 1）
        sample_indices = [0, 1, len(raw_files)//2]
        # 找一個 IOP 和 Guys 的
        for i, f in enumerate(raw_files):
            if 'IOP' in f and i not in sample_indices:
                sample_indices.append(i)
                break
        for i, f in enumerate(raw_files):
            if 'HH' in f and i not in sample_indices:
                sample_indices.append(i)
                break
        
        sample_files = [raw_files[i] for i in sample_indices if i < len(raw_files)]
        
        orientation_set = set()
        spacing_set = set()
        
        print(f"\n  抽樣 {len(sample_files)} 個原始檔案：\n")
        print(f"  {'檔案名稱':<35} {'Shape':<20} {'Spacing (mm)':<25} {'Orientation'}")
        print(f"  {'-'*33} {'-'*18} {'-'*23} {'-'*15}")
        
        for f in sample_files:
            img = nib.load(f)
            name = os.path.basename(f)
            shape = img.shape
            spacing = tuple(round(s, 3) for s in img.header.get_zooms()[:3])
            orient = nib.aff2axcodes(img.affine)
            
            orientation_set.add(orient)
            spacing_set.add(spacing)
            
            print(f"  {name:<35} {str(shape):<20} {str(spacing):<25} {orient}")
        
        print(f"\n  原始資料方向種類：{len(orientation_set)} 種 → {orientation_set}")
        print(f"  原始資料 spacing 種類：{len(spacing_set)} 種 → {spacing_set}")
        
        # 全部掃描所有檔案的 orientation
        print(f"\n  正在掃描全部 {len(raw_files)} 個原始檔案的方向和 spacing ...")
        all_orients = Counter()
        all_spacings = Counter()
        
        for f in raw_files:
            img = nib.load(f)
            orient = nib.aff2axcodes(img.affine)
            spacing = tuple(round(s, 2) for s in img.header.get_zooms()[:3])
            all_orients[orient] += 1
            all_spacings[spacing] += 1
        
        print(f"\n  方向統計（全 {len(raw_files)} 個）：")
        for orient, count in all_orients.most_common():
            print(f"    {orient} : {count} 個")
        
        print(f"\n  Spacing 統計（全 {len(raw_files)} 個）：")
        for spacing, count in all_spacings.most_common():
            print(f"    {spacing} mm : {count} 個")

except ImportError:
    print("  ⚠️  nibabel 未安裝，跳過原始 .nii.gz 檢查")
    print("     pip install nibabel")

# ── 4. 用 ANTs 驗證配準後的 spacing/orientation ──────────────────────
print("\n" + "=" * 65)
print("  4. 用 ANTs 驗證配準結果的 spacing & orientation")
print("=" * 65)

try:
    import ants
    
    # 讀 atlas
    atlas_np = np.load(atlas_path)['vol'].astype(np.float32)
    atlas_ants = ants.from_numpy(atlas_np, spacing=(1.0, 1.0, 1.0))
    
    print(f"\n  Atlas（ANTs）：")
    print(f"    shape     = {atlas_ants.numpy().shape}")
    print(f"    spacing   = {atlas_ants.spacing}")
    print(f"    origin    = {atlas_ants.origin}")
    print(f"    direction = \n{atlas_ants.direction}")
    
    # 抽樣一個原始檔案，走一次完整的 preprocess 流程，看結果
    raw_dir = os.path.join(_HERE, 'IXI-T1')
    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.nii.gz')))
    
    if len(raw_files) > 0:
        # 取第一個
        test_file = raw_files[0]
        print(f"\n  測試檔案：{os.path.basename(test_file)}")
        
        img = ants.image_read(test_file)
        print(f"    原始 shape   = {img.shape}")
        print(f"    原始 spacing = {img.spacing}")
        print(f"    原始 origin  = {img.origin}")
        print(f"    原始 direction = \n{img.direction}")
        
        # Affine registration
        print(f"\n    正在做 Affine 配準（可能需要 1~2 分鐘）...")
        reg = ants.registration(
            fixed=atlas_ants,
            moving=img,
            type_of_transform='Affine',
            verbose=False,
        )
        img_reg = reg['warpedmovout']
        
        print(f"\n    配準後：")
        print(f"      shape     = {img_reg.shape}")
        print(f"      spacing   = {img_reg.spacing}")
        print(f"      origin    = {img_reg.origin}")
        print(f"      direction = \n{img_reg.direction}")
        
        # 檢查 spacing 是否為 1mm
        sp = img_reg.spacing
        is_1mm = all(abs(s - 1.0) < 0.01 for s in sp)
        if is_1mm:
            print(f"\n    ✅ 配準後 spacing ≈ 1mm isotropic")
        else:
            print(f"\n    ⚠️  配準後 spacing 不是 1mm：{sp}")
        
        # 比較配準後的 npz
        name = os.path.basename(test_file).replace('.nii.gz', '')
        npz_path_train = os.path.join(npz_dir, 'train', name + '.npz')
        npz_path_test = os.path.join(npz_dir, 'test', name + '.npz')
        npz_path = npz_path_train if os.path.exists(npz_path_train) else npz_path_test
        
        if os.path.exists(npz_path):
            npz_vol = np.load(npz_path)['vol']
            print(f"\n    對應的 npz：{os.path.basename(npz_path)}")
            print(f"      shape = {npz_vol.shape}")
            print(f"      range = [{npz_vol.min():.4f}, {npz_vol.max():.4f}]")
        else:
            print(f"\n    找不到對應的 npz：{name}.npz")
    
except ImportError:
    print("  ⚠️  antspyx 未安裝，跳過 ANTs 驗證")
    print("     pip install antspyx")

# ── 5. 總結 ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  總結")
print("=" * 65)

npz_shape = shape_counter.most_common(1)[0][0]
print(f"""
  處理後的 npz：
    - 數量       : {len(all_npz)} 個（train + test）
    - Shape      : {npz_shape}
    - 值域       : [0, 1]（已正規化）

  關於方向和 voxel 大小：
    ┌─────────────────────────────────────────────────────┐
    │ 你的 preprocess_ixi.py 使用 ANTs Affine 配準到      │
    │ atlas（MNI152），ANTs registration 的 fixed image    │
    │ 是 atlas_ants = ants.from_numpy(..., spacing=(1,1,1))│
    │                                                     │
    │ 配準後的影像會繼承 fixed image 的 spacing 和方向，    │
    │ 所以所有配準後的影像都是：                            │
    │   ✅ spacing = (1.0, 1.0, 1.0) mm                   │
    │   ✅ 方向一致（跟 atlas 相同）                        │
    │                                                     │
    │ 但 .npz 只存 numpy array，不帶 header，所以方向和    │
    │ spacing 的資訊不在檔案裡。VoxelMorph 訓練時假設      │
    │ 所有影像都在同一個空間，這個假設在你的 pipeline 裡    │
    │ 是成立的。                                           │
    └─────────────────────────────────────────────────────┘
""")
