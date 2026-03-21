"""
VoxelMorph PyTorch 測試腳本（論文 Table I 格式）
=================================================
對 test 資料集做配準，輸出：
  - 每張影像的 Dice 均值
  - 每個解剖結構的 Dice（存成 CSV，供畫 Fig.5 boxplot 用）
  - 負 Jacobian 行列式數量與比例
  - GPU / CPU 推論時間

使用方式（在 claude_cheng 資料夾內執行）：
    python voxelmorph-code/scripts/torch/test.py --model models/0004.pt
    python voxelmorph-code/scripts/torch/test.py --model models/0004.pt --gpu -1
"""

import os
import glob
import time
import argparse
import csv
import numpy as np
import torch

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

# ============================================================
# 路徑（相對於本檔案位置，方便跨電腦部署）
# ============================================================
_HERE       = os.path.dirname(os.path.abspath(__file__))   # voxelmorph-code/scripts/torch/
_ROOT       = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))  # claude_cheng/
_DATA_DIR   = os.path.normpath(os.path.join(_HERE, '..', '..', 'data'))
_OASIS_DIR  = os.path.normpath(os.path.join(_ROOT, 'oasis'))

DEFAULT_ATLAS    = os.path.join(_DATA_DIR,  'atlas.npz')
DEFAULT_LABELS   = os.path.join(_DATA_DIR,  'labels.npz')
DEFAULT_TEST_DIR = os.path.join(_OASIS_DIR, 'oasis_npz', 'test')

# ============================================================
# seg35 連續編號 → FreeSurfer 解剖 ID 對應表
# ============================================================
SEG35_TO_FS = {
     0:  0,   1:  2,   2:  3,   3:  4,   4:  5,
     5:  7,   6:  8,   7: 10,   8: 11,   9: 12,
    10: 13,  11: 14,  12: 15,  13: 16,  14: 17,
    15: 18,  16: 26,  17: 28,  18: 30,  19: 31,
    20: 41,  21: 42,  22: 43,  23: 44,  24: 46,
    25: 47,  26: 49,  27: 50,  28: 51,  29: 52,
    30: 53,  31: 54,  32: 58,  33: 60,  34: 62,
    35: 63,
}

# FreeSurfer ID → 解剖名稱（論文 Fig.5 用）
FS_LABEL_NAMES = {
     2: 'Cerebral-WM-L',     3: 'Cerebral-Ctx-L',
     4: 'Lat-Ventricle-L',   7: 'Cereb-WM-L',
     8: 'Cereb-Ctx-L',      10: 'Thalamus-L',
    11: 'Caudate-L',        12: 'Putamen-L',
    13: 'Pallidum-L',       14: '3rd-Ventricle',
    15: '4th-Ventricle',    16: 'Brain-Stem',
    17: 'Hippocampus-L',    18: 'Amygdala-L',
    24: 'CSF',              28: 'VentralDC-L',
    31: 'Choroid-Plexus-L', 41: 'Cerebral-WM-R',
    42: 'Cerebral-Ctx-R',   43: 'Lat-Ventricle-R',
    46: 'Cereb-WM-R',       47: 'Cereb-Ctx-R',
    49: 'Thalamus-R',       50: 'Caudate-R',
    51: 'Putamen-R',        52: 'Pallidum-R',
    53: 'Hippocampus-R',    54: 'Amygdala-R',
    60: 'VentralDC-R',      63: 'Choroid-Plexus-R',
}

def remap_seg(seg):
    """seg35 連續編號 → FreeSurfer 標準 ID"""
    out = np.zeros_like(seg)
    for src, dst in SEG35_TO_FS.items():
        out[seg == src] = dst
    return out

# ============================================================
# 參數
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model',    required=True, help='模型路徑，例如 models/0004.pt')
parser.add_argument('--atlas',    default=DEFAULT_ATLAS)
parser.add_argument('--test-dir', default=DEFAULT_TEST_DIR)
parser.add_argument('--labels',   default=DEFAULT_LABELS)
parser.add_argument('--gpu',      default='0')
parser.add_argument('--out-csv',  default=None,
                    help='per-structure Dice CSV 輸出路徑（預設：模型同目錄）')
args = parser.parse_args()

# ============================================================
# 裝置設定
# ============================================================
if args.gpu and args.gpu != '-1':
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'

# CSV 輸出路徑
if args.out_csv is None:
    model_stem = os.path.splitext(os.path.basename(args.model))[0]
    csv_dir    = os.path.dirname(args.model) if os.path.dirname(args.model) else '.'
    args.out_csv = os.path.join(csv_dir, f'dice_{model_stem}.csv')

print()
print('=' * 65)
print('  VoxelMorph 測試（論文 Table I 格式）')
print('=' * 65)
print(f'  模型   : {args.model}')
print(f'  裝置   : {device}')
print(f'  CSV    : {args.out_csv}')
print()

# ============================================================
# 載入 atlas 和 labels
# ============================================================
atlas_data  = np.load(args.atlas)
atlas_vol   = atlas_data['vol']
atlas_seg   = atlas_data['seg'].astype(int)
labels_raw  = np.load(args.labels)['labels']

# 只評估 atlas 裡實際存在的標籤
atlas_unique = set(np.unique(atlas_seg).tolist())
eval_labels  = [int(l) for l in labels_raw if int(l) in atlas_unique]
label_names  = [FS_LABEL_NAMES.get(l, str(l)) for l in eval_labels]

print(f'  評估標籤數 : {len(eval_labels)}')
print()

atlas_tensor = torch.from_numpy(
    atlas_vol[np.newaxis, np.newaxis, ...]
).to(device).float()

# ============================================================
# 載入模型
# ============================================================
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()

inshape     = atlas_vol.shape
transformer = vxm.torch.layers.SpatialTransformer(inshape, mode='nearest')
transformer.to(device)

# ============================================================
# 對每張測試影像做配準
# ============================================================
test_files = sorted(glob.glob(os.path.join(args.test_dir, '*.npz')))
print(f'  測試影像數量 : {len(test_files)}')
print()
print(f'  {"Subject":<35} {"Dice":>6}  {"Neg-Jac%":>8}  {"ms":>6}')
print(f'  {"-"*35} {"-"*6}  {"-"*8}  {"-"*6}')

all_mean_dice   = []
all_per_struct  = []   # (N_subjects, N_labels)
all_neg_jac_n   = []
all_neg_jac_pct = []
all_elapsed_ms  = []

# 腦內 mask（只對腦內體素計算負 Jacobian）
brain_mask = (atlas_seg != 0)
total_brain_voxels = int(brain_mask.sum())

for fpath in test_files:
    subj = os.path.basename(fpath).replace('.npz', '')
    data = np.load(fpath)

    vol = data['vol'].astype(np.float32)
    seg = remap_seg(data['seg'].astype(int))

    vol_tensor = torch.from_numpy(
        vol[np.newaxis, np.newaxis, ...]
    ).to(device).float()

    # ------ 推論（計時）------
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        _, warp = model(vol_tensor, atlas_tensor, registration=True)

    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    all_elapsed_ms.append(elapsed_ms)

    # ------ warp 分割標籤 ------
    seg_tensor = torch.from_numpy(
        seg[np.newaxis, np.newaxis, ...].astype(np.float32)
    ).to(device)

    with torch.no_grad():
        warped_seg = transformer(seg_tensor, warp)

    warped_seg_np = warped_seg.detach().cpu().numpy().squeeze().astype(int)
    warp_np       = warp.detach().cpu().numpy().squeeze()  # (3, D, H, W)

    # ------ per-structure Dice ------
    dice_per  = vxm.py.utils.dice(warped_seg_np, atlas_seg, labels=eval_labels)
    all_per_struct.append(dice_per)
    mean_dice = float(np.mean(dice_per))
    all_mean_dice.append(mean_dice)

    # ------ 負 Jacobian（腦內體素）------
    # warp shape: (3,D,H,W) → (D,H,W,3)
    jac_det = vxm.py.utils.jacobian_determinant(warp_np.transpose(1, 2, 3, 0))
    neg_n   = int(np.sum((jac_det < 0) & brain_mask))
    neg_pct = neg_n / total_brain_voxels * 100
    all_neg_jac_n.append(neg_n)
    all_neg_jac_pct.append(neg_pct)

    print(f'  {subj:<35} {mean_dice:>6.4f}  {neg_pct:>7.3f}%  {elapsed_ms:>5.0f}')

# ============================================================
# 彙總（論文 Table I 格式）
# ============================================================
all_per_struct = np.array(all_per_struct)   # (N_subj, N_labels)

mean_dice_all  = float(np.mean(all_mean_dice))
std_dice_all   = float(np.std(all_mean_dice))
mean_neg_n     = float(np.mean(all_neg_jac_n))
std_neg_n      = float(np.std(all_neg_jac_n))
mean_neg_pct   = float(np.mean(all_neg_jac_pct))
std_neg_pct    = float(np.std(all_neg_jac_pct))
mean_sec       = float(np.mean(all_elapsed_ms)) / 1000
std_sec        = float(np.std(all_elapsed_ms)) / 1000

print()
print('=' * 75)
print('  TABLE I 格式')
print('=' * 75)
hdr = f'  {"Method":<28} {"Dice":>16}  {"GPU/CPU sec":>12}  {"|Jφ|≤0":>14}  {"% |Jφ|≤0":>14}'
print(hdr)
print(f'  {"-"*28} {"-"*16}  {"-"*12}  {"-"*14}  {"-"*14}')

model_name = f'VoxelMorph ({os.path.basename(args.model)})'
row = (f'  {model_name:<28} '
       f'{mean_dice_all:.3f} ({std_dice_all:.3f})  '
       f'  {mean_sec:.3f} ({std_sec:.3f})  '
       f' {mean_neg_n:.0f} ({std_neg_n:.0f})  '
       f' {mean_neg_pct:.3f} ({std_neg_pct:.3f})')
print(row)
print('=' * 75)

# 論文數字對照
print()
print('  論文對照（IEEE TMI 2019, Table I）')
print(f'  {"Affine only":<28} 0.584 (0.157)  {"—":>12}  {"0":>14}  {"0.000":>14}')
print(f'  {"ANTs SyN (CC)":<28} 0.749 (0.136)  {"—":>12}  {"9662 (6258)":>14}  {"0.185 (0.091)":>14}')
print(f'  {"VoxelMorph (CC)":<28} 0.753 (0.145)  {"0.45 (0.01)":>12}  {"19077 (5928)":>14}  {"0.366 (0.114)":>14}')
print(f'  {"VoxelMorph (MSE)":<28} 0.752 (0.140)  {"0.45 (0.01)":>12}  {"9606 (4516)":>14}  {"0.184 (0.087)":>14}')
print()

# ============================================================
# 儲存 per-structure CSV（供畫 Fig.5 boxplot）
# ============================================================
with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['subject'] + label_names)
    for i, fpath in enumerate(test_files):
        subj = os.path.basename(fpath).replace('.npz', '')
        writer.writerow([subj] + [f'{v:.6f}' for v in all_per_struct[i]])
    # 最後附上統計
    writer.writerow(['MEAN'] + [f'{np.mean(all_per_struct[:, j]):.6f}'
                                 for j in range(len(eval_labels))])
    writer.writerow(['STD']  + [f'{np.std(all_per_struct[:, j]):.6f}'
                                 for j in range(len(eval_labels))])

print(f'  per-structure Dice → {args.out_csv}')
print()
print(f'  下一步，產生論文 Fig.4+5+6 圖表：')
print(f'  python draw-img/visualize_registration.py --model {args.model} --csv {args.out_csv}')
print()
