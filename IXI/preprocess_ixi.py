"""
IXI T1 前處理腳本
流程：N4 bias correction → 去顱骨 → 對位到 atlas 空間 → 正規化 → 打包 npz

用法：
    python preprocess_ixi.py
    python preprocess_ixi.py --save-nii   # 額外輸出 .nii.gz 供驗證方向

輸出：
    IXI/IXI_preprocessed/train/  (90%)
    IXI/IXI_preprocessed/test/   (10%)
    IXI/IXI_preprocessed/nii/    (若有 --save-nii，抽樣輸出 .nii.gz)
"""

import os
import sys
import glob
import random
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--in-dir',   default=os.path.join(_HERE, 'IXI-T1'),
                    help='原始 .nii.gz 資料夾')
parser.add_argument('--out-dir',  default=os.path.join(_HERE, 'IXI_preprocessed'),
                    help='輸出資料夾')
parser.add_argument('--atlas',    default=os.path.join(_HERE, 'atlas_mni152_09c_resize.npz'),
                    help='對位目標（atlas.npz 或 .nii.gz），建議用已 resize 到 target-shape 的版本')
parser.add_argument('--target-shape', default='192,224,192',
                    help='輸出影像大小，預設 192,224,192（必須能被 16 整除）')
parser.add_argument('--skip-done', action='store_true', default=True,
                    help='略過已處理的檔案（中斷後可續跑）')
parser.add_argument('--no-brain-extract', action='store_true', default=False,
                    help='跳過去顱骨（測試用）')
parser.add_argument('--save-nii', type=int, default=0, metavar='N',
                    help='額外輸出前 N 筆 .nii.gz 檔案，用來驗證方向和 spacing（預設 0 = 不輸出）')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# ── 檢查套件 ─────────────────────────────────────────────────────────
try:
    import ants
except ImportError:
    print("❌ 請先安裝 antspyx：pip install antspyx")
    sys.exit(1)

try:
    import antspynet
    HAS_ANTSPYNET = True
except ImportError:
    HAS_ANTSPYNET = False
    if not args.no_brain_extract:
        print("⚠️  找不到 antspynet，將改用簡易閾值去顱骨")
        print("   如需精確去顱骨：pip install antspynet\n")

# ── 設定 ─────────────────────────────────────────────────────────────
random.seed(args.seed)
target_shape = tuple(int(x) for x in args.target_shape.split(','))

# 建立輸出資料夾
for split in ['train', 'test']:
    os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)

if args.save_nii > 0:
    nii_dir = os.path.join(args.out_dir, 'nii')
    os.makedirs(nii_dir, exist_ok=True)
    nii_saved_count = 0
    print(f"[save-nii] 將額外輸出前 {args.save_nii} 筆 .nii.gz 到 {nii_dir}/")

# ── 載入 atlas 當對位目標 ────────────────────────────────────────────
atlas_path = os.path.normpath(args.atlas)
print(f"載入 atlas：{atlas_path}")

if atlas_path.endswith('.npz'):
    atlas_np = np.load(atlas_path)['vol'].astype(np.float32)
    atlas_ants = ants.from_numpy(atlas_np, spacing=(1.0, 1.0, 1.0))
else:
    atlas_ants = ants.image_read(atlas_path)

# atlas 大小必須與 target_shape 一致
if atlas_ants.numpy().shape != target_shape:
    print(f"❌ Atlas shape {atlas_ants.numpy().shape} != target_shape {target_shape}")
    print(f"   請先用 make_atlas.py --target-shape {','.join(map(str, target_shape))} 產生正確大小的 atlas")
    print(f"   或指定 --atlas 指向已 resize 的 atlas（如 atlas_mni152_09c_resize.npz）")
    sys.exit(1)

print(f"Atlas shape：{atlas_ants.numpy().shape}")
print()

# ── 列出所有輸入檔案並分割 ───────────────────────────────────────────
all_files = sorted(glob.glob(os.path.join(args.in_dir, '*.nii.gz')))
if len(all_files) == 0:
    print(f"❌ 找不到 .nii.gz 檔案：{args.in_dir}")
    sys.exit(1)

random.shuffle(all_files)
n = len(all_files)
n_train = int(n * 0.90)

split_map = {}
for i, f in enumerate(all_files):
    if i < n_train: split_map[f] = 'train'
    else:           split_map[f] = 'test'

train_c = sum(1 for v in split_map.values() if v == 'train')
test_c  = sum(1 for v in split_map.values() if v == 'test')
print(f"資料分割：train={train_c}  test={test_c}  (共 {n} 筆)")
print()

# ── 簡易閾值去顱骨（antspynet 不存在時的備用） ──────────────────────
def simple_skull_strip(img_ants):
    arr = img_ants.numpy()
    thr = arr.max() * 0.15
    mask_arr = (arr > thr).astype(np.float32)
    # 用 ANTs morphological close 填洞
    mask_ants = img_ants.new_image_like(mask_arr)
    mask_ants = ants.morphological(mask_ants, radius=3, operation='close')
    return ants.mask_image(img_ants, mask_ants)

# ── 主處理迴圈 ───────────────────────────────────────────────────────
ok_count   = 0
skip_count = 0
fail_count = 0

for idx, src_path in enumerate(all_files):
    name  = os.path.basename(src_path).replace('.nii.gz', '')
    split = split_map[src_path]
    dst_path = os.path.join(args.out_dir, split, name + '.npz')

    # 略過已處理
    if args.skip_done and os.path.exists(dst_path):
        skip_count += 1
        print(f"[{idx+1:3d}/{n}] ⏭  略過（已存在）：{name}")
        continue

    print(f"[{idx+1:3d}/{n}] 處理：{name}  ({split})")

    try:
        # 1. 讀影像
        img = ants.image_read(src_path)
        print(f"        原始大小：{img.shape}  spacing：{img.spacing}")

        # 2. N4 bias field correction
        print("        N4 bias correction ...")
        img_n4 = ants.n4_bias_field_correction(img)

        # 3. 去顱骨
        if args.no_brain_extract:
            img_brain = img_n4
        elif HAS_ANTSPYNET:
            print("        去顱骨（antspynet）...")
            prob = antspynet.brain_extraction(img_n4, modality='t1', verbose=False)
            mask = ants.threshold_image(prob, 0.5, 1.0)
            img_brain = ants.mask_image(img_n4, mask)
        else:
            print("        去顱骨（簡易閾值）...")
            img_brain = simple_skull_strip(img_n4)

        # 4. 對位到 atlas（Affine）
        print("        Affine 對位到 atlas ...")
        reg = ants.registration(
            fixed   = atlas_ants,
            moving  = img_brain,
            type_of_transform = 'Affine',
            verbose = False,
        )
        img_reg = reg['warpedmovout']

        # 5. 確認輸出 shape（若 atlas 已 resize 到 target_shape，這裡應直接一致）
        img_np = img_reg.numpy().astype(np.float32)
        if img_np.shape != target_shape:
            raise ValueError(
                f"配準輸出 shape {img_np.shape} != target {target_shape}。\n"
                f"請確認 atlas 已 resize 到 {target_shape}：\n"
                f"  python IXI\\make_atlas.py --target-shape {','.join(map(str, target_shape))} ..."
            )

        # 6. 灰值正規化到 [0, 1]（clip 1% / 99% percentile 後再正規化）
        p1, p99 = np.percentile(img_np[img_np > 0], [1, 99])
        img_np  = np.clip(img_np, p1, p99)
        img_np  = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np  = img_np.astype(np.float32)

        # 7. 存成 npz
        np.savez_compressed(dst_path, vol=img_np)
        ok_count += 1
        print(f"        ✓ 儲存：{dst_path}  shape={img_np.shape}  "
              f"min={img_np.min():.3f} max={img_np.max():.3f}")

        # 8. 額外存 .nii.gz（用 ANTs 配準後的 image 物件，保留 header）
        if args.save_nii > 0 and nii_saved_count < args.save_nii:
            import nibabel as nib
            # 方法 A：從 ANTs image 直接寫（保留配準後的 spacing/direction）
            nii_ants_path = os.path.join(nii_dir, name + '_ants.nii.gz')
            ants.image_write(img_reg, nii_ants_path)
            # 方法 B：從 npz 的 numpy array 寫（用 identity affine，模擬 VoxelMorph 讀取的方式）
            nii_npz_path = os.path.join(nii_dir, name + '_npz.nii.gz')
            nii_img = nib.Nifti1Image(img_np, np.diag([1.0, 1.0, 1.0, 1.0]))
            nib.save(nii_img, nii_npz_path)
            nii_saved_count += 1
            print(f"        ✓ [save-nii {nii_saved_count}/{args.save_nii}] "
                  f"{nii_ants_path} (有header)")
            print(f"          {nii_npz_path} (identity affine)")

    except Exception as e:
        fail_count += 1
        print(f"        ❌ 失敗：{e}")
        import traceback
        traceback.print_exc()

    print()

# ── 完成摘要 ─────────────────────────────────────────────────────────
print("=" * 55)
print(f"完成！  成功={ok_count}  略過={skip_count}  失敗={fail_count}")
print(f"輸出資料夾：{args.out_dir}")
