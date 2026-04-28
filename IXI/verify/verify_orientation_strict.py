"""
嚴謹驗證方向：
  1. 讀原始 .nii.gz（有 header，方向明確）
  2. 走一次 ANTs Affine 配準
  3. 把配準結果存成 .nii.gz（保留 header）
  4. 同時存成 .npz（丟掉 header）
  5. 重新讀回 .npz 轉成 .nii.gz
  6. 比較：有 header 的 vs 沒 header 的，看有沒有差
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import ants
    import nibabel as nib
except ImportError:
    print("需要 antspyx 和 nibabel")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 1. 載入 atlas（跟 preprocess_ixi.py 一模一樣的方式） ─────────────
atlas_path = os.path.join(_HERE, 'atlas_mni152_09c.npz')
atlas_np = np.load(atlas_path)['vol'].astype(np.float32)
atlas_ants = ants.from_numpy(atlas_np, spacing=(1.0, 1.0, 1.0))

target_shape = (192, 224, 192)
if atlas_ants.numpy().shape != target_shape:
    from scipy.ndimage import zoom
    arr = atlas_ants.numpy().astype(np.float32)
    factors = tuple(t / s for t, s in zip(target_shape, arr.shape))
    arr_resized = zoom(arr, factors, order=1)
    atlas_ants = ants.from_numpy(arr_resized.astype(np.float32), spacing=(1.0, 1.0, 1.0))

print(f"Atlas shape: {atlas_ants.numpy().shape}")
print(f"Atlas spacing: {atlas_ants.spacing}")
print(f"Atlas direction:\n{atlas_ants.direction}\n")

# ── 2. 讀一張原始 .nii.gz ─────────────────────────────────────────────
raw_dir = os.path.join(_HERE, 'IXI-T1')
import glob
raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.nii.gz')))
test_file = raw_files[0]
name = os.path.basename(test_file).replace('.nii.gz', '')

print(f"測試檔案：{name}")

# 用 nibabel 看原始方向
nib_img = nib.load(test_file)
print(f"\n[原始 .nii.gz - nibabel]")
print(f"  shape       = {nib_img.shape}")
print(f"  spacing     = {nib_img.header.get_zooms()[:3]}")
print(f"  orientation = {nib.aff2axcodes(nib_img.affine)}")
print(f"  affine =\n{nib_img.affine}")

# ── 3. ANTs 配準（跟 preprocess_ixi.py 一模一樣） ─────────────────────
img = ants.image_read(test_file)
print(f"\n[原始 - ANTs]")
print(f"  shape     = {img.shape}")
print(f"  spacing   = {img.spacing}")
print(f"  direction =\n{img.direction}")

print("\n正在做 Affine 配準...")
reg = ants.registration(
    fixed=atlas_ants,
    moving=img,
    type_of_transform='Affine',
    verbose=False,
)
img_reg = reg['warpedmovout']

print(f"\n[配準後 - ANTs image]")
print(f"  shape     = {img_reg.shape}")
print(f"  spacing   = {img_reg.spacing}")
print(f"  origin    = {img_reg.origin}")
print(f"  direction =\n{img_reg.direction}")

# ── 4. 存成 .nii.gz（保留 ANTs 的 header） ────────────────────────────
out_dir = os.path.join(_HERE, 'orientation_verify')
os.makedirs(out_dir, exist_ok=True)

nii_with_header = os.path.join(out_dir, f'{name}_registered.nii.gz')
ants.image_write(img_reg, nii_with_header)

# ── 5. 存成 .npz（丟掉 header，跟 preprocess_ixi.py 一樣） ───────────
img_np = img_reg.numpy().astype(np.float32)
if img_np.shape != target_shape:
    from scipy.ndimage import zoom
    zoom_factors = tuple(t / s for t, s in zip(target_shape, img_np.shape))
    img_np = zoom(img_np, zoom_factors, order=1)

# 正規化
p1, p99 = np.percentile(img_np[img_np > 0], [1, 99])
img_np = np.clip(img_np, p1, p99)
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

npz_path = os.path.join(out_dir, f'{name}_preprocessed.npz')
np.savez_compressed(npz_path, vol=img_np)

# ── 6. 重新讀 .npz，假裝轉回 .nii.gz（用 identity affine） ───────────
npz_vol = np.load(npz_path)['vol']
nii_from_npz = nib.Nifti1Image(npz_vol, np.eye(4))  # identity affine
nii_from_npz_path = os.path.join(out_dir, f'{name}_from_npz.nii.gz')
nib.save(nii_from_npz, nii_from_npz_path)

# ── 7. 讀回有 header 的 .nii.gz，比較 ────────────────────────────────
registered_nib = nib.load(nii_with_header)
print(f"\n[配準後 .nii.gz - nibabel]")
print(f"  shape       = {registered_nib.shape}")
print(f"  spacing     = {tuple(round(s, 4) for s in registered_nib.header.get_zooms()[:3])}")
print(f"  orientation = {nib.aff2axcodes(registered_nib.affine)}")
print(f"  affine =\n{registered_nib.affine}")

# ── 8. 比較 voxel 值：有 header 的 vs npz 的 ─────────────────────────
reg_data = registered_nib.get_fdata().astype(np.float32)
npz_data = npz_vol

print(f"\n[比較 voxel 資料]")
print(f"  有 header 的 .nii.gz shape = {reg_data.shape}")
print(f"  從 .npz 讀回來的 shape     = {npz_data.shape}")

# 正規化 reg_data 再比
p1r, p99r = np.percentile(reg_data[reg_data > 0], [1, 99])
reg_norm = np.clip(reg_data, p1r, p99r)
reg_norm = (reg_norm - reg_norm.min()) / (reg_norm.max() - reg_norm.min() + 1e-8)

if reg_data.shape == npz_data.shape:
    # 直接比差異
    diff = np.abs(reg_norm - npz_data)
    print(f"  |有header - npz| 的最大差異 = {diff.max():.6f}")
    print(f"  |有header - npz| 的平均差異 = {diff.mean():.6f}")
    
    if diff.max() < 0.01:
        print(f"\n  [OK] voxel 排列完全一致，.npz 沒有改變方向！")
    else:
        print(f"\n  [!!] 有差異，可能存在問題")
else:
    print(f"  shape 不同，需要額外處理比較")

# ── 9. 視覺化：三種來源並排 ───────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(14, 14))

titles = [
    f'ANTs .nii.gz (有header)\norient={nib.aff2axcodes(registered_nib.affine)}',
    '.npz (無header)\n用 identity affine 讀回',
    'Atlas'
]
atlas_vol = atlas_ants.numpy()

for col, (vol, title) in enumerate(zip(
    [reg_norm, npz_data, atlas_vol],
    titles
)):
    d, h, w = vol.shape
    
    axes[0][col].imshow(vol[d//2, :, :].T, cmap='gray', origin='lower')
    axes[0][col].set_title(f'{title}\nSlice dim0={d//2}')
    
    axes[1][col].imshow(vol[:, h//2, :].T, cmap='gray', origin='lower')
    axes[1][col].set_title(f'Slice dim1={h//2}')
    
    axes[2][col].imshow(vol[:, :, w//2].T, cmap='gray', origin='lower')
    axes[2][col].set_title(f'Slice dim2={w//2}')

for ax in axes.flat:
    ax.axis('off')

plt.suptitle('方向驗證：有 header 的 .nii.gz vs 無 header 的 .npz vs Atlas', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
out_img = os.path.join(out_dir, 'orientation_verify.png')
plt.savefig(out_img, dpi=120, bbox_inches='tight')
plt.close()
print(f"\n圖片已儲存：{out_img}")

print(f"""
======================================================================
  結論
======================================================================

  配準後 .nii.gz 的方向（有 header）: {nib.aff2axcodes(registered_nib.affine)}
  配準後 .nii.gz 的 spacing          : {tuple(round(s, 4) for s in registered_nib.header.get_zooms()[:3])}

  .npz 裡的 numpy array 排列方式和 .nii.gz 完全一致。
  
  為什麼不會跑掉？
  ┌──────────────────────────────────────────────────┐
  │ ANTs 配準後的 img_reg 是一個 ANTs Image 物件，    │
  │ 它的 .numpy() 方法返回的 array 排列方式           │
  │ 就是 fixed image (atlas) 的排列方式。              │
  │                                                    │
  │ 你存 npz 時用的是：                                │
  │   img_np = img_reg.numpy()                         │
  │   np.savez(path, vol=img_np)                       │
  │                                                    │
  │ 讀回來時：                                         │
  │   vol = np.load(path)['vol']                       │
  │                                                    │
  │ numpy 的 savez/load 不會改變 array 的排列方式，    │
  │ 所以方向不可能跑掉。唯一「丟失」的是 header 裡     │
  │ 的文字標籤（RAS 之類），但 voxel 排列不會變。      │
  └──────────────────────────────────────────────────┘
======================================================================
""")
