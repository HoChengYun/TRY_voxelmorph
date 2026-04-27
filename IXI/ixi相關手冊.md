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

### 3.3 用 make_atlas.py 製作 atlas npz

腳本路徑：`IXI/make_atlas.py`

```powershell
# 建議直接加 --target-shape，產出可以直接進 VoxelMorph 的 atlas
python IXI\make_atlas.py `
    --t1   IXI\mni_icbm152_nlin_asym_09c_nifti\mni_icbm152_t1_tal_nlin_asym_09c.nii `
    --mask IXI\mni_icbm152_nlin_asym_09c_nifti\mni_icbm152_t1_tal_nlin_asym_09c_mask.nii `
    --target-shape 192,224,192 `
    --save-nii

# 輸出：
#   IXI/atlas_mni152_09c.npz       ← (192, 224, 192)  min/max: 0.0 / 1.0
#   IXI/atlas_mni152_09c.nii.gz    ← 同內容的 NIfTI 版（--save-nii 產生，驗證用）
```

腳本做了什麼：

1. 用 `ants.mask_image()` 套 mask，去掉頭骨
2. clip 1%~99% percentile 去掉極端值
3. 正規化到 [0, 1]
4. resize 到 target-shape（`scipy.ndimage.zoom`, **雙線性插值**，不是補零也不是裁切）
5. 存成 `.npz`（key: `vol`）

| 參數 | 說明 |
|------|------|
| `--t1` | MNI152 T1 .nii 路徑 |
| `--mask` | brain mask .nii 路徑 |
| `--out` | 輸出 .npz 路徑（預設 `atlas_mni152_09c.npz`）|
| `--target-shape` | resize 目標，如 `192,224,192`（必須能被 16 整除）|
| `--save-nii` | 額外輸出 .nii.gz，可用 ITK-SNAP 驗證方向和 spacing |

> **重要**：建議一定要加 `--target-shape`，讓 atlas 的 shape 直接與 preprocess_ixi.py 的輸出一致，避免後續任何 resize 問題。

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

# 執行前處理（使用已 resize 的 atlas）
python IXI\preprocess_ixi.py `
    --atlas IXI\atlas_mni152_09c_resize.npz `
    --save-nii 3
```

每張影像的處理步驟：

1. N4 Bias Field Correction（修正亮度不均）
2. 去顱骨（antspynet 精確版，或簡易閾值備用）
3. Affine 對位到 MNI152 atlas（ANTs registration，輸出直接繼承 atlas 的 shape 和 spacing）
4. Shape 驗證（必須與 atlas 一致，否則報錯）
5. 灰值正規化到 [0, 1]（clip 1%~99% percentile）
6. 存成 `.npz`（key: `vol`）

> **注意**：之前的版本在 step 3 之後有一個 `scipy.ndimage.zoom` 做 resize（因為舊 atlas 是 193×229×193，配準後需要 zoom 到 192×224×192）。**現在改用已 resize 的 atlas**，ANTs 配準輸出直接就是 `(192, 224, 192)`，不再需要任何 zoom。如果 atlas shape 與 target_shape 不一致，腳本會直接報錯。

輸出資料夾結構：

```
IXI/IXI_preprocessed/
  train/   ← 522 筆（90%）
  test/    ← 59 筆（10%）
  nii/     ← 驗證用 .nii.gz（若有 --save-nii）
```

---

### 4.5 前處理腳本參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--in-dir` | `IXI/IXI-T1` | 原始 .nii.gz 資料夾 |
| `--out-dir` | `IXI/IXI_preprocessed` | 輸出資料夾 |
| `--atlas` | `IXI/atlas_mni152_09c_resize.npz` | 對位目標，**必須已 resize 到 target-shape** |
| `--target-shape` | `192,224,192` | 輸出影像大小（必須能被 16 整除）|
| `--skip-done` | `True`（預設開啟） | 略過已處理的檔案，中斷後可續跑 |
| `--no-brain-extract` | `False` | 跳過去顱骨（測試用） |
| `--save-nii` | `0`（不輸出） | 額外輸出前 N 筆 .nii.gz，用來驗證方向 |
| `--seed` | `42` | 亂數種子，確保 train/test 分割可重現 |

> **`--seed` 說明**：固定了 `random.shuffle` 的亂數種子，確保每次跑腳本都得到一樣的 train/test 分割，保證實驗可重現性（reproducibility）。
>
> **`--save-nii` 說明**：會在 `IXI_preprocessed/nii/` 輸出兩種 .nii.gz：
> - `XXX_ants.nii.gz`：ANTs 配準後直接寫出（有完整 header：spacing、direction）
> - `XXX_npz.nii.gz`：從 npz numpy array 用 identity affine 寫出
>
> 用 ITK-SNAP 開兩個對比，即可 100% 確認 .npz 的方向沒有跑掉。

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

用 `make_atlas.py --target-shape 192,224,192` 在製作 atlas 時就 resize 好：

```powershell
python IXI\make_atlas.py `
    --t1   IXI\mni_icbm152_nlin_asym_09c_nifti\mni_icbm152_t1_tal_nlin_asym_09c.nii `
    --mask IXI\mni_icbm152_nlin_asym_09c_nifti\mni_icbm152_t1_tal_nlin_asym_09c_mask.nii `
    --target-shape 192,224,192 `
    --save-nii
```

### Resize 方法說明

| 方法 | 做法 | 適用情境 |
|------|------|----------|
| **雙線性插值（zoom, order=1）** ✅ 目前使用 | 整體均勻縮放，每個 voxel 重新計算 | 尺寸微調（差距 < 5%），對稱不變 |
| 補零（zero-padding） | 在邊緣填 0 | ❌ 會導致腦部不對稱（單邊補零） |
| 裁切（cropping） | 砍掉邊緣 voxel | ❌ 可能砍到腦組織邊緣 |

本次 resize：`(193, 229, 193)` → `(192, 224, 192)`，zoom factors ≈ `(0.995, 0.978, 0.995)`，每個軸只縮了不到 2.2%，幾乎看不出差異。

### 流程圖

```
make_atlas.py --target-shape 192,224,192
  MNI152 (193,229,193) → zoom → atlas_resize.npz (192,224,192)
                                      ↓
preprocess_ixi.py --atlas atlas_resize.npz
  IXI原始 (256,256,150) → ANTs Affine(fixed=atlas) → 直接輸出 (192,224,192)
                                                      ↓
                                                  不需要任何 zoom
                                                      ↓
                                                  正規化 → .npz
```

> **以前的流程**：atlas 沒有先 resize，preprocess_ixi.py 裡面有兩個地方做 zoom（載入 atlas 時 + 配準後安全檢查）。
> **現在的流程**：atlas 先 resize 好，preprocess_ixi.py 裡面零次 zoom，如果 shape 不一致會直接報錯。

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
| 配準後 spacing | ✅ `(1.0, 1.0, 1.0)` mm isotropic |
| 配準後 direction | ✅ 單位矩陣（Identity） |
| 有 header 的 .nii.gz vs .npz 的 voxel 差異 | ✅ 最大差異 = 0.000000 |
| 視覺化（3 家醫院 × 5 張 vs atlas） | ✅ 方向完全一致 |

---

## 7. 資料夾結構總覽

```
claude_cheng/
├── IXI/
│   ├── IXI-T1/                        ← 原始 IXI T1 .nii.gz（581 張）
│   ├── mni_icbm152_nlin_asym_09c_nifti/  ← 下載的 MNI152 2009c NIfTI
│   ├── atlas_mni152_09c.npz           ← make_atlas.py 產生（原始大小 193×229×193）
│   ├── atlas_mni152_09c_resize.npz    ← resize 後版本（192×224×192）← 訓練用這個
│   ├── make_atlas.py                  ← 製作 atlas npz（支援 --save-nii）
│   ├── preprocess_ixi.py              ← IXI 前處理腳本（支援 --save-nii N）
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

## 工具

### `read_nii_header.py`

```powershell
# 讀單個檔案
python IXI\read_nii_header.py  IXI\orientation_verify\IXI002-Guys-0828-T1_registered.nii.gz

# 讀多個
python IXI\read_nii_header.py  IXI\orientation_verify\*.nii.gz

# 顯示完整 header
python IXI\read_nii_header.py  --full  IXI\orientation_verify\*.nii.gz

```

