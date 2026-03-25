# VoxelMorph × IXI：新資料集前處理與訓練實作手冊

何承運 · 2026

---

## 1. 背景知識回顧

### 1.1 atlas.npz 裡面有什麼

VoxelMorph 訓練用的 atlas 檔案（atlas.npz）包含三個 key：

| Key | 形狀 / 說明 |
|-----|-------------|
| `vol` | `(160, 192, 224)` 腦部 T1 影像，灰值正規化至 [0, 0.73] |
| `seg` | `(160, 192, 224)` FreeSurfer 解剖標籤（39 個唯一值，使用 FreeSurfer ID） |
| `train_avg` | `(256,)` 訓練集各標籤的平均 Dice，供 test.py 比較用 |

> **重點**：訓練時只讀 `vol`，`seg` 靜靜躺在 npz 裡不會被使用。`seg` 只在 `test.py` 評估 Dice 時才被載入。

---

### 1.2 VoxelMorph 為何是非監督式學習

Loss 函數只有兩項，完全不依賴人工標註：

```
Loss = NCC(Warped_vol, Atlas_vol)  +  λ · Smooth(φ)

NCC    → Normalized Cross-Correlation，衡量配準後影像與 Atlas 的灰值相似度
Smooth → 懲罰形變場梯度，確保形變平滑合理
```

| 比較項目 | 監督式 | VoxelMorph（非監督） |
|----------|--------|---------------------|
| 訓練需要 | 配對影像 + ground truth 形變場 | 只需要腦影像本身 |
| ground truth 來源 | 人工標註或傳統方法預算 | 不需要 |
| Seg 角色 | 可能參與訓練 | 只在測試時評估用 |

---

### 1.3 Scan-to-Atlas 訓練時 seg 完全不參與

train.py 第 70 行，atlas 只讀 vol：

```python
# train.py 第 70 行
atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol', ...)

# scan_to_atlas generator 也沒有傳 return_segs=True
gen = volgen(vol_names, batch_size=batch_size, **kwargs)  # seg 不載入
```

完整流程中 seg 的角色：

- **打包 npz 時**：vol + seg 都存進去
- **訓練時**：只讀 vol，seg 靜靜躺在 npz 裡沒被碰
- **測試時**：才讀 seg，套形變場 φ 去算 Dice

---

## 2. 沒有 FreeSurfer 分割時的驗證方法

當新資料集（如 IXI）沒有 FreeSurfer seg 時，可以用以下方式驗證配準品質：

| 方法 | 說明 | 需要 seg？ |
|------|------|-----------|
| NCC / MSE / SSIM | 配準後 Warped 和 Atlas 的灰值相似度 | 否 |
| 負 Jacobian 比例 | 形變場是否有折疊（越低越好） | 否 |
| SynthSeg 產生偽標籤 | 用深度學習快速產生 seg 後算 Dice | 否（自動產生） |
| FreeSurfer 重跑 | 最標準，但每張要 6~10 小時 | 否（自動產生） |

> **本次選擇**：IXI 資料採用影像相似度（NCC / SSIM）作為驗證指標，不需要 seg，直接對位到 MNI152 即可。

### SynthSeg 快速產生 seg（備用）

若之後需要 seg，SynthSeg 是最省事的工具（幾分鐘 / 張）：

```bash
# 需要 FreeSurfer 環境
mri_synthseg --i input.nii.gz --o output_seg.nii.gz
```

---

## 3. MNI152 2009c Atlas 準備

### 3.1 為什麼選 ICBM152 2009c Asymmetric

| 版本比較 | 說明 |
|----------|------|
| ICBM152 6th generation（舊版） | 2006 年，舊論文常用，無現成去顱骨版本 |
| ICBM152 2009c Symmetric | 左右強制對稱，適合族群平均分析 |
| **ICBM152 2009c Asymmetric（本次）** | 保留真實左右不對稱，適合個體配準 ✓ |

> **下載來源**：https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/ → 選 NIFTI 版本

---

### 3.2 下載後的檔案結構

```
mni_icbm152_t1_tal_nlin_asym_09c.nii          ← T1 含頭骨
mni_icbm152_t1_tal_nlin_asym_09c_mask.nii     ← 腦部 mask（用這個去顱骨）
mni_icbm152_t1_tal_nlin_asym_09c_eye_mask.nii
mni_icbm152_t1_tal_nlin_asym_09c_face_mask.nii

原始 shape: (193, 229, 193)  spacing: 1mm isotropic
```

> ⚠️ **注意**：193 和 229 都無法被 16 整除，無法直接進入 VoxelMorph 訓練，需 resize。

---

### 3.3 用 make_atlas.py 製作 atlas_mni152_09c.npz

腳本路徑：`IXI/make_atlas.py`

```powershell
python IXI\make_atlas.py \
    --t1   你的路徑\mni_icbm152_t1_tal_nlin_asym_09c.nii \
    --mask 你的路徑\mni_icbm152_t1_tal_nlin_asym_09c_mask.nii

# 輸出：IXI/atlas_mni152_09c.npz
# shape: (193, 229, 193)  min/max: 0.0 / 1.0
```

腳本做了什麼：

- 用 `ants.mask_image()` 套 mask，去掉頭骨
- clip 1%~99% percentile 去掉極端值
- 正規化到 [0, 1]
- 存成 `.npz`（key: `vol`）

---

## 4. IXI T1 前處理流程

### 4.1 安裝套件

```bash
# 啟動虛擬環境
.\vxm_env\Scripts\activate

pip install antspyx      # 提供 N4、去顱骨、Affine 配準
pip install antspynet    # 深度學習去顱骨模型（精確版）

# antspynet 第一次執行時會自動從 figshare 下載預訓練模型（約 200MB）
```

---

### 4.2 ANTsPy / ANTsPyNet 是什麼

**ANTsPy（antspyx）** 是 ANTs（Advanced Normalization Tools）的 Python 版本。ANTs 是神經影像最常用的配準工具，VoxelMorph 論文中用於比較的 ANTs SyN-CC 就是它。主要功能：

- N4 Bias Field Correction（修正 MRI 亮度不均）
- Affine / SyN 影像配準
- 影像重採樣、mask 操作

**ANTsPyNet** 是 ANTsPy 的深度學習擴充套件，提供精確的腦部去顱骨模型：

- `brain_extraction(img, modality='t1')`：輸出腦部機率 mask
- 底層使用 TensorFlow，Windows 原生版不支援 GPU（用 CPU 跑）

---

### 4.3 影像大小限制

VoxelMorph 的 U-Net 有 4 層 downsampling，每層除以 2，因此三個維度都必須能被 16 整除：

```
160 ÷ 16 = 10  ✓      192 ÷ 16 = 12  ✓      224 ÷ 16 = 14  ✓
193 ÷ 16 = 12.06 ✗    229 ÷ 16 = 14.31 ✗    150 ÷ 2 = 75（奇數）✗
```

| 目標大小 | GPU 記憶體估計 | 說明 |
|----------|---------------|------|
| `192 × 224 × 192` | ~9 GB（緊） | 最接近 MNI152 原始，細節保留最多 |
| `160 × 192 × 224` | ~7 GB（安全） | OASIS 標準大小，與論文比較方便 |
| `160 × 192 × 160` | ~5 GB（省） | 較節省記憶體，解析度略低 |

> **本次設定**：使用 `192,224,192`（跟 MNI152 原始最接近），8GB GPU 可跑。

---

### 4.4 執行前處理腳本

腳本路徑：`IXI/preprocess_ixi.py`

```powershell
# 先刪掉舊的處理結果（若有）
Remove-Item -Recurse -Force IXI\IXI_preprocessed

# 執行前處理
python IXI\preprocess_ixi.py \
    --atlas IXI\atlas_mni152_09c.npz \
    --target-shape 192,224,192
```

每張影像的處理步驟：

1. N4 Bias Field Correction（修正亮度不均）
2. 去顱骨（antspynet 精確版，或簡易閾值備用）
3. Affine 對位到 MNI152 atlas（標準化空間）
4. Resize 到目標大小（`scipy.ndimage.zoom`）
5. 灰值正規化到 [0, 1]（clip 1%~99% percentile）
6. 存成 `.npz`（key: `vol`）

輸出資料夾結構：

```
IXI/IXI_preprocessed/
  train/   ← 522 筆（90%）
  test/    ← 59 筆（10%）
```

---

### 4.5 前處理腳本參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--in-dir` | `IXI/IXI-T1` | 原始 .nii.gz 資料夾 |
| `--out-dir` | `IXI/IXI_preprocessed` | 輸出資料夾 |
| `--atlas` | `IXI/atlas_mni152_09c.npz` | 對位目標（npz 或 nii/nii.gz） |
| `--target-shape` | `192,224,192` | 輸出影像大小 |
| `--skip-done` | `True`（預設開啟） | 略過已處理的檔案，中斷後可續跑 |
| `--no-brain-extract` | `False` | 跳過去顱骨（測試用） |
| `--seed` | `42` | 亂數種子，確保 train/test 分割可重現 |

> **`--seed` 說明**：固定了 `random.shuffle` 的亂數種子，確保每次跑腳本都得到一樣的 train/test 分割，保證實驗可重現性（reproducibility）。

---

## 5. 用 IXI 重新訓練 VoxelMorph

### 5.1 訓練指令

```powershell
python voxelmorph-code\scripts\torch\train.py \
    --datadir  IXI\IXI_preprocessed\train \
    --atlas    IXI\atlas_mni152_09c.npz \
    --model-dir models\ixi_mni \
    --epochs   200 \
    --gpu      0
```

---

### 5.2 與 OASIS 模型的差異

| 比較項目 | OASIS 模型 | IXI 新模型 |
|----------|-----------|------------|
| 資料集 | 414 位受試者（老年/失智） | 581 位健康成人 |
| Atlas 空間 | OASIS 訓練集平均腦 | MNI152 2009c Asymmetric（標準） |
| 影像大小 | 160 × 192 × 224 | 192 × 224 × 192（依設定） |
| 評估指標 | Dice（seg35 標籤） | NCC / SSIM（無 seg） |

---

### 5.3 測試指令（影像相似度）

```powershell
python voxelmorph-code\scripts\torch\test.py \
    --model   models\ixi_mni\XXXX.pt \
    --datadir IXI\IXI_preprocessed\test \
    --atlas   IXI\atlas_mni152_09c.npz \
    --gpu     0
```

> ⚠️ **注意**：IXI 資料沒有 seg，test.py 無法計算 Dice。若需要 Dice，可先用 SynthSeg 或 `antspynet.desikan_killiany_tourville_labeling()` 產生偽標籤。

---

## 6. 常見問題

| 問題 | 原因 / 解法 |
|------|-------------|
| `visualize_registration.py` 卡住不動 | 沒有加 `--gpu 0`，預設走 CPU，3D U-Net 推論很慢。加 `--gpu 0` 即可。 |
| `RuntimeError: size XXX not divisible` | 影像大小無法被 16 整除。在 preprocess_ixi.py 指定 `--target-shape` 修正。 |
| `CUDA out of memory` | 影像太大或 batch 太多。縮小 `--target-shape` 或加 `--batch-size 1`。 |
| antspynet TF GPU warning（Windows） | TensorFlow >= 2.11 在 Windows 原生不支援 GPU，用 CPU 跑。不影響結果，可忽略。 |
| `tf.function retracing` warning | 各張影像大小略不同，TF 重新編譯計算圖。不影響結果，可忽略。 |
| `--skip-done` 要怎麼關掉 | 加 `--no-skip-done`，或直接刪掉 IXI_preprocessed 資料夾重跑。 |
| `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 192 but got size 193` | atlas npz 的 shape 和訓練影像大小不一致，見下方「Atlas Resize 問題」。 |

---

## 6.1 Atlas Resize 問題（重要）

### 問題描述

`make_atlas.py` 產生的 `atlas_mni152_09c.npz` 原始 shape 為 `(193, 229, 193)`，但 `preprocess_ixi.py` 輸出的訓練影像是 `(192, 224, 192)`。訓練時 `train.py` 把 atlas vol 直接當 target，兩者大小不一致，導致 `torch.cat` 時報錯：

```
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 192 but got size 193 for tensor number 1 in the list.
```

### 根本原因

`train.py` 沒有自動 resize 邏輯，它假設「atlas 和訓練影像一定是同樣大小」，大小要在前處理階段就對齊好。

### 解法一：一次性修正現有 npz（快速）

```python
python -c "
import numpy as np
from scipy.ndimage import zoom

data = np.load('IXI/atlas_mni152_09c.npz')
vol = data['vol'].astype(np.float32)
print('原始 shape:', vol.shape)

target = (192, 224, 192)
factors = tuple(t / s for t, s in zip(target, vol.shape))
vol_resized = zoom(vol, factors, order=1)
print('resize 後:', vol_resized.shape)

np.savez_compressed('IXI/atlas_mni152_09c_resize.npz', vol=vol_resized)
print('Done')
"
```

### 解法二：make_atlas.py 加 --target-shape（根本解）

`make_atlas.py` 已加入 `--target-shape` 參數，之後直接輸出 resize 好的版本：

```powershell
python IXI\make_atlas.py `
    --t1   你的路徑\mni_icbm152_t1_tal_nlin_asym_09c.nii `
    --mask 你的路徑\mni_icbm152_t1_tal_nlin_asym_09c_mask.nii `
    --target-shape 192,224,192
```

### 為什麼 IXI train/test 資料不用重跑

`preprocess_ixi.py` 流程是：
1. 用 atlas 做 Affine 對位（對齊腦部空間，參考原始 193,229,193）
2. resize 到 `--target-shape`（輸出 192,224,192）

第一步只是配準用的參考座標系，輸出的 npz 已經是 `(192, 224, 192)`，跟 atlas 原始大小無關，所以不需要重跑前處理。

---

## 7. 資料夾結構總覽

```
claude_cheng/
├── IXI/
│   ├── IXI-T1/                    ← 原始 IXI T1 .nii.gz（581 張）
│   ├── mni_icbm152_nl_asym_09c/   ← 下載的 MNI152 2009c NIfTI
│   ├── atlas_mni152_09c.npz       ← make_atlas.py 產生
│   ├── make_atlas.py              ← 製作 atlas npz
│   ├── preprocess_ixi.py          ← IXI 前處理腳本
│   └── IXI_preprocessed/
│       ├── train/  （522 筆）
│       └── test/   （59 筆）
├── models/
│   ├── exp1/                      ← OASIS 訓練結果
│   └── ixi_mni/                   ← IXI 訓練結果（待）
└── voxelmorph-code/
    ├── data/atlas.npz             ← OASIS 原始 atlas
    └── scripts/torch/
        ├── train.py
        ├── test.py
        └── batch_test.py
```
