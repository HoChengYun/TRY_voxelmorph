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
parser.add_argument('--out-dir',  required=True,
                    help='輸出資料夾（必填）')
parser.add_argument('--atlas',    required=True,
                    help='對位目標（.nii.gz 帶 header，建議用 make_atlas.py 產生的版本）')
parser.add_argument('--target-shape', default='192,224,192',
                    help='輸出影像大小，預設 192,224,192（必須能被 16 整除）')
parser.add_argument('--skip-done', action='store_true', default=True,
                    help='略過已處理的檔案（中斷後可續跑）')
parser.add_argument('--no-brain-extract', action='store_true', default=False,
                    help='跳過去顱骨（測試用）')
parser.add_argument('--save-nii', action='store_true', default=False,
                    help='額外輸出每筆的 .nii.gz 檔案到 nii/ 資料夾，用來驗證方向和 spacing')
parser.add_argument('--vis', type=str, default=None, metavar='FILE',
                    help='視覺化模式：指定一個 .nii.gz，畫出每步處理結果到單張圖（不會存 npz）')
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

# 建立輸出資料夾（--vis 模式不建 train/test）
if not args.vis:
    for split in ['train', 'test']:
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)
else:
    os.makedirs(args.out_dir, exist_ok=True)

if args.save_nii:
    nii_dir = os.path.join(args.out_dir, 'nii')
    os.makedirs(nii_dir, exist_ok=True)
    print(f"[save-nii] 將額外輸出所有 .nii.gz 到 {nii_dir}/")

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

# --vis 模式：只處理指定的那一張
if args.vis:
    vis_file = os.path.normpath(args.vis)
    if not os.path.exists(vis_file):
        print(f"❌ 找不到 --vis 指定的檔案：{vis_file}")
        sys.exit(1)
    all_files = [vis_file]
    if vis_file not in split_map:
        split_map[vis_file] = 'vis'
    print(f"\n[vis] 視覺化模式：只處理 {os.path.basename(vis_file)}")

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

        # 7. 存成 npz（--vis 模式不存）
        if not args.vis:
            np.savez_compressed(dst_path, vol=img_np)
            ok_count += 1
            print(f"        ✓ 儲存：{dst_path}  shape={img_np.shape}  "
                  f"min={img_np.min():.3f} max={img_np.max():.3f}")

        # 8. 額外存 .nii.gz（正規化後的值 + img_reg 的完整 header）
        if args.save_nii:
            # 用 img_reg 的 header（spacing, origin, direction）+ 正規化後的值
            img_nii = img_reg.new_image_like(img_np)
            nii_path = os.path.join(nii_dir, name + '.nii.gz')
            ants.image_write(img_nii, nii_path)
            print(f"        ✓ [nii] {nii_path}")

        # 9. 視覺化模式：畫出每步結果 + overlay + header 比較
        if args.vis:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import nibabel as nib

            # 取 header 資訊
            def get_header_info(ants_img):
                direc = ants_img.direction
                aff = np.eye(4)
                aff[:3, :3] = direc * np.array(ants_img.spacing)
                aff[:3, 3] = ants_img.origin
                orient = nib.aff2axcodes(aff)
                return {
                    'shape': ants_img.shape,
                    'spacing': tuple(round(s, 4) for s in ants_img.spacing),
                    'origin': tuple(round(o, 2) for o in ants_img.origin),
                    'orient': orient,
                }

            # 取中間切片 + header 資訊
            def get_step_info(ants_img, title, is_np=False):
                if is_np:
                    vol = ants_img
                    info = f'{title}\nshape={vol.shape}'
                else:
                    vol = ants_img.numpy()
                    h = get_header_info(ants_img)
                    info = f"{title}\n{h['shape']}  sp={h['spacing']}\norient={h['orient']}"
                d, h, w = vol.shape
                return {
                    'title': info,
                    'slice0': vol[d//2, :, :],
                    'slice1': vol[:, h//2, :],
                    'slice2': vol[:, :, w//2],
                }

            # ── 圖 1：每步流程 ────────────────────────────────
            steps = [
                get_step_info(img, '1. Original'),
                get_step_info(img_n4, '2. N4 Correction'),
                get_step_info(img_brain, '3. Skull Strip'),
                get_step_info(img_reg, '4. Affine Reg'),
                get_step_info(img_np, '5. Normalize [0,1]', is_np=True),
                get_step_info(atlas_ants, 'Atlas (target)'),
            ]

            views = ['slice0', 'slice1', 'slice2']
            view_labels = ['Axis 0 (mid)', 'Axis 1 (mid)', 'Axis 2 (mid)']
            n_steps = len(steps)
            fig, axes = plt.subplots(3, n_steps, figsize=(3.5 * n_steps, 10))

            for col, step in enumerate(steps):
                for row, view in enumerate(views):
                    sl = step[view]
                    axes[row][col].imshow(sl.T, cmap='gray', origin='lower')
                    axes[row][col].axis('off')
                    if row == 0:
                        axes[row][col].set_title(step['title'], fontsize=8)
                    if col == 0:
                        axes[row][col].set_ylabel(view_labels[row], fontsize=10)

            fig.suptitle(f'Preprocessing Pipeline: {name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            vis_path = os.path.join(args.out_dir, f'vis_{name}_pipeline.png')
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"        ✓ [vis] {vis_path}")

            # ── 圖 2：Atlas vs 處理後 overlay ─────────────────
            atlas_np_vis = atlas_ants.numpy()
            fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
            slice_labels = ['Axis 0 (mid)', 'Axis 1 (mid)', 'Axis 2 (mid)']

            for col in range(3):
                d, h, w = img_np.shape
                if col == 0:
                    sl_proc = img_np[d//2, :, :]
                    sl_atlas = atlas_np_vis[d//2, :, :]
                elif col == 1:
                    sl_proc = img_np[:, h//2, :]
                    sl_atlas = atlas_np_vis[:, h//2, :]
                else:
                    sl_proc = img_np[:, :, w//2]
                    sl_atlas = atlas_np_vis[:, :, w//2]

                # 上排：並排比較
                axes2[0][col].imshow(sl_atlas.T, cmap='gray', origin='lower')
                axes2[0][col].set_title(f'Atlas - {slice_labels[col]}', fontsize=10)
                axes2[0][col].axis('off')

                # 下排：Atlas（紅色）疊在處理後（灰階）上面
                p_norm = sl_proc / (sl_proc.max() + 1e-8)
                a_norm = sl_atlas / (sl_atlas.max() + 1e-8)
                alpha = 0.7
                gray = p_norm.T
                a_t = a_norm.T
                # Atlas 紅色在最上層（atlas 越亮紅越深）
                rgb = np.stack([
                    gray * (1 - alpha * a_t) + a_t * alpha,  # R
                    gray * (1 - alpha * a_t),                 # G
                    gray * (1 - alpha * a_t),                 # B
                ], axis=-1)
                rgb = np.clip(rgb, 0, 1)
                axes2[1][col].imshow(rgb, origin='lower')
                axes2[1][col].set_title(f'Atlas (red, top) + Processed (gray)', fontsize=9)
                axes2[1][col].axis('off')

            fig2.suptitle(f'Atlas vs Processed: {name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            overlay_path = os.path.join(args.out_dir, f'vis_{name}_overlay.png')
            plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"        ✓ [vis] {overlay_path}")

            # ── Header 比較表 ─────────────────────────────────
            h_orig = get_header_info(img)
            h_atlas = get_header_info(atlas_ants)
            h_proc = get_header_info(img_reg)

            print(f"\n{'='*70}")
            print(f"  Header 比較")
            print(f"{'='*70}")
            print(f"  {'':18s} {'原始影像':20s} {'處理後影像':20s} {'Atlas':20s}")
            print(f"  {'-'*16} {'-'*18} {'-'*18} {'-'*18}")
            print(f"  {'Shape':18s} {str(h_orig['shape']):20s} {str(h_proc['shape']):20s} {str(h_atlas['shape']):20s}")
            print(f"  {'Spacing (mm)':18s} {str(h_orig['spacing']):20s} {str(h_proc['spacing']):20s} {str(h_atlas['spacing']):20s}")
            print(f"  {'Origin':18s} {str(h_orig['origin']):20s} {str(h_proc['origin']):20s} {str(h_atlas['origin']):20s}")
            print(f"  {'Orientation':18s} {str(h_orig['orient']):20s} {str(h_proc['orient']):20s} {str(h_atlas['orient']):20s}")

            # 檢查處理後和 atlas 是否一致
            match_shape = h_proc['shape'] == h_atlas['shape']
            match_sp = h_proc['spacing'] == h_atlas['spacing']
            match_orient = h_proc['orient'] == h_atlas['orient']
            print(f"\n  處理後 vs Atlas：")
            print(f"    Shape 一致       : {'✅' if match_shape else '❌'}")
            print(f"    Spacing 一致     : {'✅' if match_sp else '❌'}")
            print(f"    Orientation 一致 : {'✅' if match_orient else '❌'}")
            print(f"{'='*70}")

            print(f"\n視覺化完成，只處理 1 張後結束。")
            break  # --vis 模式只處理 1 張

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
