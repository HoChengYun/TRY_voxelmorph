"""
OASIS neurite-oasis.v1.0 資料前處理腳本
=========================================
將 .nii.gz 批次轉換成 VoxelMorph 訓練用的 .npz 格式
並自動切分 train / test 資料集

使用方式（在 oasis/ 資料夾內執行）：
    python prepare_oasis.py

或從任何位置執行：
    python oasis/prepare_oasis.py

輸出結構：
    oasis/oasis_npz/
    ├── train/   (374 筆，含 vol + seg)
    └── test/    (40 筆，含 vol + seg)
"""

import os
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ============================================================
# 路徑（相對於本檔案位置，方便跨電腦部署）
# ============================================================
_HERE       = os.path.dirname(os.path.abspath(__file__))  # oasis/

OASIS_DIR    = os.path.join(_HERE, 'neurite-oasis.v1.0')
OUTPUT_DIR   = os.path.join(_HERE, 'oasis_npz')
SUBJECTS_TXT = os.path.join(OASIS_DIR, 'subjects.txt')

TRAIN_RATIO = 0.9                           # 90% train, 10% test
RANDOM_SEED = 42

# 要使用的檔案
VOL_FILE = 'aligned_norm.nii.gz'           # 預處理腦影像
SEG_FILE = 'aligned_seg35.nii.gz'          # 35 標籤分割

# ============================================================
# 讀取受試者清單
# ============================================================
with open(SUBJECTS_TXT, 'r') as f:
    subjects = [line.strip() for line in f if line.strip()]

print(f'總受試者數：{len(subjects)}')

# 隨機切分 train / test
random.seed(RANDOM_SEED)
random.shuffle(subjects)

n_test  = int(len(subjects) * (1 - TRAIN_RATIO))
n_train = len(subjects) - n_test

train_subjects = subjects[n_test:]
test_subjects  = subjects[:n_test]

print(f'Train：{n_train} 筆')
print(f'Test ：{n_test}  筆')
print()

# ============================================================
# 建立輸出資料夾
# ============================================================
train_dir = os.path.join(OUTPUT_DIR, 'train')
test_dir  = os.path.join(OUTPUT_DIR, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir,  exist_ok=True)

# ============================================================
# 轉換函式
# ============================================================
def convert_subject(subject_id, output_dir):
    subj_path = os.path.join(OASIS_DIR, subject_id)
    vol_path  = os.path.join(subj_path, VOL_FILE)
    seg_path  = os.path.join(subj_path, SEG_FILE)

    # 檢查檔案存在
    if not os.path.exists(vol_path):
        print(f'  ⚠ 找不到：{vol_path}，跳過')
        return False
    if not os.path.exists(seg_path):
        print(f'  ⚠ 找不到：{seg_path}，跳過')
        return False

    # 載入 NIfTI
    vol = nib.load(vol_path).get_fdata().squeeze().astype(np.float32)
    seg = nib.load(seg_path).get_fdata().squeeze().astype(np.int32)

    # 確認值域（aligned_norm 已歸一化，但再確認一次）
    if vol.max() > 1.0:
        vol = vol / vol.max()

    # 儲存為 .npz（含 vol 和 seg）
    out_path = os.path.join(output_dir, f'{subject_id}.npz')
    np.savez_compressed(out_path, vol=vol, seg=seg)
    return True


# ============================================================
# 批次轉換 Train
# ============================================================
print('轉換 Train 資料...')
train_ok = 0
for subj in tqdm(train_subjects, ncols=70):
    if convert_subject(subj, train_dir):
        train_ok += 1

# ============================================================
# 批次轉換 Test
# ============================================================
print('轉換 Test 資料...')
test_ok = 0
for subj in tqdm(test_subjects, ncols=70):
    if convert_subject(subj, test_dir):
        test_ok += 1

# ============================================================
# 驗證其中一個輸出檔案
# ============================================================
print()
print('驗證輸出...')
sample_file = os.path.join(train_dir, f'{train_subjects[0]}.npz')
sample = np.load(sample_file)
print(f'  範例檔案  : {os.path.basename(sample_file)}')
print(f'  vol shape : {sample["vol"].shape}')
print(f'  vol range : [{sample["vol"].min():.4f}, {sample["vol"].max():.4f}]')
print(f'  seg shape : {sample["seg"].shape}')
print(f'  seg labels: {np.unique(sample["seg"])}')

# ============================================================
# 完成
# ============================================================
print()
print('=' * 50)
print(f'  轉換完成！')
print(f'  Train：{train_ok} 筆  →  {train_dir}')
print(f'  Test ：{test_ok}  筆  →  {test_dir}')
print()
print('  下一步，執行訓練（從 claude_cheng 根目錄執行）：')
print()
print('  python voxelmorph-code/scripts/torch/train.py \\')
print('      oasis/oasis_npz/train \\')
print('      --atlas voxelmorph-code/data/atlas.npz \\')
print('      --model-dir models/ \\')
print('      --gpu 0 \\')
print('      --image-loss ncc \\')
print('      --lambda 1.0')
print('=' * 50)
