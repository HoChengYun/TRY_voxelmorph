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

### 3.3 用 make_atlas.py 製作 atlas

腳本路徑：`IXI/make_atlas.py`

```powershell
python IXI\make_atlas.py `
    --t1   IXI\mni_icbm152_nlin_asym_09c_nifti\mni_icbm152_t1_tal_nlin_asym_09c.nii `
    --mask IXI\mni_icbm152_nlin_asym_09c_nifti\mni_icbm152_t1_tal_nlin_asym_09c_mask.nii `
    --target-shape 192,224,192

# 同時輸出兩個檔案：
#   IXI/atlas_mni152_09c.nii.gz  ← 帶 MNI152 header（給 preprocess_ixi.py 的 ANTs 配準用）
#   IXI/atlas_mni152_09c.npz    ← 只有 numpy array（給 VoxelMorph train.py 用）
```

腳本做了什麼：

1. 用 `ants.mask_image()` 套 mask，去掉頭骨
2. clip 1%~99% percentile 去掉極端值
3. 正規化到 [0, 1]
4. resize 到 target-shape（`ants.resample_image`，**保留 MNI152 header**）
5. 同時存成 `.nii.gz`（帶 header）和 `.npz`（只有 array）

| 參數 | 說明 |
|------|------|
| `--t1` | MNI152 T1 .nii 路徑 |
| `--mask` | brain mask .nii 路徑 |
| `--out` | 輸出路徑（不含副檔名，自動產生 .npz + .nii.gz）|
| `--target-shape` | resize 目標，如 `192,224,192`（必須能被 16 整除）|

> **為什麼用 `ants.resample_image` 而不是 `scipy.ndimage.zoom`？**
> `ants.resample_image` 會保留原始 MNI152 的 header（origin、direction），讓 ANTs 配準時能用真正的空間資訊做初始化。`scipy.zoom` 只處理 numpy array，header 會丟失。
>
> **Spacing 不是精確 1mm**：resize 後 spacing ≈ `(1.005, 1.022, 1.005)` mm，因為 `ants.resample_image` 保留了物理範圍（FOV），193mm / 192 voxels ≈ 1.005mm。差距 < 2.3%，對訓練無影響。

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
# 執行前處理（必填 --out-dir 和 --atlas）
python IXI\preprocess_ixi.py `
    --out-dir IXI\IXI_preprocessed `
    --atlas   IXI\atlas_mni152_09c.nii.gz `
    --save-nii

# 視覺化單張前處理流程（不會存 npz，不建 train/test）
python IXI\preprocess_ixi.py `
    --out-dir IXI\preprocess_vis `
    --atlas   IXI\atlas_mni152_09c.nii.gz `
    --vis     IXI\IXI-T1\IXI002-Guys-0828-T1.nii.gz
```

每張影像的處理步驟：

1. N4 Bias Field Correction（修正亮度不均）
2. 去顱骨（antspynet 精確版，或簡易閾值備用）
3. Affine 對位到 MNI152 atlas（ANTs registration，**使用 atlas 和受試者的 header 初始化**）
4. Shape 驗證（必須與 atlas 一致，否則報錯）
5. 灰值正規化到 [0, 1]（clip 1%~99% percentile）
6. 存成 `.npz`（key: `vol`）

> **ANTs 如何使用 header**：`ants.registration()` 接收 ANTs Image 物件，會讀取兩邊的 spacing、origin、direction，在物理空間（mm 座標）做對齊。用帶真正 MNI152 header 的 .nii.gz atlas，比用 identity header 的 .npz atlas 初始化更精準。

輸出資料夾結構：

```
IXI/IXI_preprocessed/
  train/   ← 522 筆（90%）
  test/    ← 59 筆（10%）
  nii/     ← 每筆的 .nii.gz（若有 --save-nii）
```

---

### 4.5 前處理腳本參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--in-dir` | `IXI/IXI-T1` | 原始 .nii.gz 資料夾 |
| `--out-dir` | **必填** | 輸出資料夾 |
| `--atlas` | **必填** | 對位目標（.nii.gz 帶 header，用 make_atlas.py 產生）|
| `--target-shape` | `192,224,192` | 輸出影像大小（必須能被 16 整除）|
| `--skip-done` | `True`（預設開啟） | 略過已處理的檔案，中斷後可續跑 |
| `--no-brain-extract` | `False` | 跳過去顱骨（測試用） |
| `--save-nii` | `False` | 額外輸出**每筆** .nii.gz（正規化後值 + 完整 header）|
| `--vis FILE` | `None` | 視覺化模式：指定一個 .nii.gz，畫出每步流程圖 + overlay 圖 |
| `--seed` | `42` | 亂數種子，確保 train/test 分割可重現 |

> **`--save-nii` 說明**：會在 `nii/` 輸出每筆的 `.nii.gz`，內容是正規化 0-1 後的值（和 .npz 一致），帶有 ANTs 配準後的完整 header（spacing, origin, direction）。可用 `read_nii_header.py` 或 ITK-SNAP 驗證方向和 spacing。
>
> **`--vis` 說明**：視覺化模式，只處理指定的 1 張，產生兩張圖：
> - `vis_XXX_pipeline.png`：6 欄（Original → N4 → Skull Strip → Affine → Normalize → Atlas）× 3 切面，每欄顯示 shape/spacing/orientation
> - `vis_XXX_overlay.png`：Atlas（紅色）疊在處理後影像（灰階）上，確認對齊效果
> - 並在 terminal 印出 Header 比較表（原始 vs 處理後 vs Atlas）
>
> `--vis` 模式不會存 .npz，不會建 train/test 資料夾。

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

### 5.3 test_ixi.py — 單次測試（NCC / SSIM）

IXI 沒有 seg，原本的 `test.py` 讀 `atlas['seg']` 會直接報錯。改用專門為 IXI 寫的 `test_ixi.py`：

```powershell
python voxelmorph-code\scripts\torch\test_ixi.py `
    --model    models\exp2_IXI\0100.pt `
    --atlas    IXI\atlas_mni152_09c_resize.npz `
    --test-dir IXI\IXI_preprocessed\test `
    --gpu      0
```

輸出每張的 NCC / SSIM，並自動存成 CSV（`models/exp2_IXI/eval_0100.csv`）。

| 參數 | 說明 |
|------|------|
| `--model` | 模型路徑（.pt） |
| `--atlas` | atlas npz（resize 後版本） |
| `--test-dir` | test 資料夾 |
| `--out-csv` | 指定 CSV 輸出路徑（預設：模型同目錄） |
| `--gpu` | GPU ID，`-1` 表示 CPU |

**注意事項：**
- 需在最前面加 `os.environ['NEURITE_BACKEND'] = 'pytorch'` 避免 TF Keras 版本衝突
- 需在 `model.eval()` 前加 `model.to(device)` 確保模型在 GPU 上

---

### 5.4 batch_test_ixi.py — 逐 epoch 比較曲線

跑 `models/` 資料夾裡所有 `.pt`，畫出 NCC / SSIM vs Epoch 曲線：

```powershell
python voxelmorph-code\scripts\torch\batch_test_ixi.py `
    --model-dir models\exp2_IXI `
    --atlas     IXI\atlas_mni152_09c_resize.npz `
    --test-dir  IXI\IXI_preprocessed\test `
    --out-dir   draw-img\output `
    --step      10 `
    --gpu       0
```

| 參數 | 說明 |
|------|------|
| `--model-dir` | 存放 .pt 的資料夾 |
| `--step` | 每幾個 epoch 評估一次（預設 1，全跑；建議 10 節省時間） |
| `--out-dir` | 輸出圖片和 CSV 的資料夾 |

輸出：
- `draw-img/output/epoch_curve.png`：NCC / SSIM vs Epoch 折線圖
- `draw-img/output/epoch_curve.csv`：各 epoch 數值

---

### 5.5 visualize_reg_ixi.py — 配準視覺化四格圖

輸出論文常見的四格圖：Source / Atlas / Warped / Difference

```powershell
python draw-img\visualize_reg_ixi.py `
    --model    models\exp2_IXI\0100.pt `
    --atlas    IXI\atlas_mni152_09c_resize.npz `
    --test-dir IXI\IXI_preprocessed\test `
    --out-dir  draw-img\output `
    --gpu      0
```

| 參數 | 說明 |
|------|------|
| `--subject` | 指定單張 npz 路徑，不指定則從 test-dir 隨機選 |
| `--slice-axis` | 切面方向：`axial`（預設）/ `coronal` / `sagittal` |
| `--out-dir` | 輸出圖片資料夾 |

Difference 圖顯示資訊：
- colorbar：顏色對應差異絕對值大小（0~1）
- `MAD`（Mean Absolute Difference）：切面平均差異，越接近 0 越好
- `max`：切面最大差異點數值

> **注意**：Atlas / Warped 顯示時會自動 flipud 對齊 Source 的左右方向（因為 imshow 用 `.T` 轉置，flipud 在顯示上才等於左右翻轉）。

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

MNI152 2009c 原始 shape 為 `(193, 229, 193)`，無法被 16 整除，不能直接進 VoxelMorph U-Net。

### 解法（現行做法）

`make_atlas.py --target-shape 192,224,192` 用 `ants.resample_image` resize，**同時產生 .nii.gz（帶 header）和 .npz（給 train.py）**。

### Resize 方法說明

| 方法 | 做法 | 適用情境 |
|------|------|----------|
| **`ants.resample_image`** ✅ 目前使用 | 均勻縮放 + 保留 header | 保留 MNI152 空間資訊，供 ANTs 配準初始化 |
| `scipy.ndimage.zoom` | 均勻縮放，丟失 header | 之前的做法，可行但 ANTs 初始化不精準 |
| 補零（zero-padding） | 在邊緣填 0 | ❌ 會導致腦部不對稱 |
| 裁切（cropping） | 砍掉邊緣 voxel | ❌ 可能砍到腦組織邊緣 |

**Spacing 說明**：resize 後 spacing ≈ `(1.005, 1.022, 1.005)` mm（不是精確 1mm），因為保留了原始物理範圍（193mm / 192 voxels ≈ 1.005mm）。差距 < 2.3%，對訓練無影響。

### 流程圖

```
make_atlas.py --target-shape 192,224,192
  MNI152 (193,229,193) @ 1mm
    → ants.resample_image（保留 header）
    → atlas.nii.gz (192,224,192) @ ~1.005mm  ← 帶 MNI152 header
    → atlas.npz    (192,224,192)             ← 只有 numpy array
                        ↓
preprocess_ixi.py --atlas atlas.nii.gz
  ANTs 讀取 atlas header（origin, direction, spacing）
  ANTs 讀取受試者 header
  → 用兩邊 header 初始化 → 優化 Affine → 輸出 (192,224,192)
  → 正規化 → .npz
```

---

## 6.2 方向驗證（.npz 沒有 header）

### 為什麼要驗證

`.npz` 只存 numpy array，不帶方向/spacing 等 header 資訊。擔心方向是否在處理過程中跑掉。

### 為什麼不會跑掉

1. **ANTs 配準**：所有影像都以同一個 atlas 當 fixed image，輸出一定在 atlas 空間裡
2. **`img_reg.numpy()`**：ANTs image 的 `.numpy()` 返回的 array 排列方式就是 fixed image 的排列方式
3. **`np.savez` / `np.load`**：不會改變 array 的排列方式（不會旋轉、翻轉）

### 驗證方法

#### 方法 A：用 `--save-nii` 參數（推薦）

直接在前處理流程中額外輸出 .nii.gz，100% 保證和 .npz 來自同一段 code：

```powershell
python IXI\preprocess_ixi.py --save-nii 3
```

輸出兩種 .nii.gz 到 `IXI_preprocessed/nii/`：
- `XXX_ants.nii.gz`：ANTs 配準後直接寫出（有 header）
- `XXX_npz.nii.gz`：從 .npz 的 numpy array 用 identity affine 寫出

用 ITK-SNAP 或 3D Slicer 比對即可確認方向一致。

#### 方法 B：用 `make_atlas.py --save-nii`

```powershell
python IXI\make_atlas.py --save-nii ...
```

atlas 的 .nii.gz 版本可以用 ITK-SNAP 開啟確認方向。

### 驗證結果（2026/04/27 已驗證）

| 檢查項目 | 結果 |
|----------|------|
| 581 個 .npz shape 是否一致 | ✅ 全部 `(192, 224, 192)` |
| 原始 IXI 方向是否一致 | ✅ 全部 581 個都是 `('P', 'S', 'R')` |
| 配準後 spacing | ✅ ≈ `(1.005, 1.022, 1.005)` mm（保留物理範圍） |
| 配準後 direction | ✅ Identity（與 atlas 一致） |
| 有 header 的 .nii.gz vs .npz 的 voxel 差異 | ✅ 最大差異 = 0.000000 |
| 視覺化（3 家醫院 × 5 張 vs atlas） | ✅ 方向完全一致 |

---

## 7. 資料夾結構總覽

```
claude_cheng/
├── IXI/
│   ├── IXI-T1/                        ← 原始 IXI T1 .nii.gz（581 張）
│   ├── mni_icbm152_nlin_asym_09c_nifti/  ← 下載的 MNI152 2009c NIfTI
│   ├── atlas_mni152_09c.nii.gz        ← make_atlas.py 產生（帶 header，給 ANTs 配準用）
│   ├── atlas_mni152_09c.npz           ← make_atlas.py 產生（只有 array，給 train.py 用）
│   ├── make_atlas.py                  ← 製作 atlas（同時輸出 .nii.gz + .npz）
│   ├── preprocess_ixi.py              ← IXI 前處理腳本（支援 --save-nii N）
│   ├── read_nii_header.py             ← 讀取 NIfTI header 工具
│   ├── verify_orientation.py          ← 方向視覺化驗證（抽樣畫圖）
│   ├── verify_orientation_strict.py   ← 方向嚴謹驗證（有header vs 無header 比對）
│   ├── verify_preprocess.py           ← 前處理結果統計驗證
│   └── IXI_preprocessed/
│       ├── train/  （522 筆）
│       ├── test/   （59 筆）
│       └── nii/    （--save-nii 輸出的驗證用 .nii.gz）
├── models/
│   ├── exp1/                          ← OASIS 訓練結果
│   └── ixi_mni/                       ← IXI 訓練結果（待）
├── draw-img/
│   ├── visualize_reg_ixi.py           ← 配準四格視覺化圖（Source/Atlas/Warped/Diff）
│   └── output/                        ← 輸出的圖片和 CSV
└── voxelmorph-code/
    ├── data/atlas.npz                 ← OASIS 原始 atlas
    └── scripts/torch/
        ├── train.py
        ├── test.py                    ← OASIS 用（需要 seg）
        ├── test_ixi.py                ← IXI 用，計算 NCC / SSIM
        └── batch_test_ixi.py          ← 逐 epoch 評估，畫曲線圖
```

## 8. 工具腳本

### read_nii_header.py — 讀取 NIfTI header

路徑：`IXI/read_nii_header.py`

顯示 .nii / .nii.gz 的 shape、spacing、orientation、affine、voxel size、origin、direction。支援任意數量檔案，多檔時自動顯示總結表。

```powershell
# 讀單個檔案
python IXI\read_nii_header.py  IXI\atlas_mni152_09c.nii.gz

# 讀多個（支援 wildcard）
python IXI\read_nii_header.py  IXI\IXI_preprocessed\nii\*.nii.gz

# 顯示完整 header
python IXI\read_nii_header.py  --full  IXI\atlas_mni152_09c.nii.gz
```

