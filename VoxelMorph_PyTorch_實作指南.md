# VoxelMorph PyTorch 腦部 MRI 配準實作指南

> 本指南記錄從零開始跑通 VoxelMorph 的完整過程，包含踩過的坑與修正方法。
> 所有指令皆在 `claude_cheng/` 根目錄執行。

---

## 目錄

1. [什麼是 VoxelMorph](#1-什麼是-voxelmorph)
2. [資料夾結構](#2-資料夾結構)
3. [環境安裝](#3-環境安裝)
4. [資料集準備（OASIS）](#4-資料集準備oasis)
5. [訓練](#5-訓練)
6. [測試與評估](#6-測試與評估)
7. [視覺化配準結果](#7-視覺化配準結果)
8. [模型原理](#8-模型原理)
9. [常見問題與解決方案](#9-常見問題與解決方案)
10. [參考資源](#10-參考資源)

---

## 1. 什麼是 VoxelMorph

VoxelMorph 是一個**基於深度學習的醫學影像配準框架**。

傳統配準方法（ANTs、Elastix）需要對每對影像反覆迭代優化，每次要花數分鐘到數小時。VoxelMorph 改為**訓練一個 U-Net，讓它直接預測形變場**，訓練完成後對新影像配準只需不到一秒。

**核心論文：**
- CVPR 2018 / IEEE TMI 2019: *VoxelMorph: A Learning Framework for Deformable Medical Image Registration*
- MICCAI 2018: *Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration*

**我們使用的配置：**
- 後端：PyTorch
- 資料集：neurite-oasis.v1.0（414 個腦部 MRI）
- 訓練模式：Scan-to-Atlas（每張影像都配準到同一個標準腦）
- 評估指標：Dice 係數（衡量配準後解剖結構重疊程度）

---

## 2. 資料夾結構

```
claude_cheng/                          ← 工作根目錄（所有指令從這裡執行）
│
├── voxelmorph-code/                   ← VoxelMorph 原始碼庫
│   ├── data/
│   │   ├── atlas.npz                  ← 標準腦模板（160×192×224，含 vol + seg）
│   │   └── labels.npz                 ← 30 個評估用解剖標籤的 FreeSurfer ID
│   ├── scripts/torch/
│   │   ├── train.py                   ← 訓練腳本
│   │   ├── register.py                ← 單對影像配準推論
│   │   └── test.py                    ← 批次測試，計算 Dice（我們自己寫的）
│   └── voxelmorph/torch/
│       ├── networks.py                ← VxmDense 主模型
│       ├── losses.py                  ← NCC, MSE, Grad, Dice 損失函數
│       └── layers.py                  ← SpatialTransformer, VecInt
│
├── oasis/                             ← OASIS 資料集相關
│   ├── neurite-oasis.v1.0/            ← 原始 NIfTI 資料（414 個受試者）
│   ├── oasis_npz/
│   │   ├── train/                     ← 訓練資料（374 個 .npz）
│   │   └── test/                      ← 測試資料（40 個 .npz）
│   └── prepare_oasis.py               ← 將 .nii.gz 轉為 .npz 的腳本
│
├── draw-img/                          ← 視覺化相關
│   ├── visualize_registration.py      ← 畫配準結果圖
│   └── registration_result.png        ← 輸出圖片
│
├── models/                            ← 訓練存檔
│   ├── 0001.pt  0002.pt  ...          ← 每個 epoch 的模型
│   └── （數字 = 訓練到第幾個 epoch）
│
├── vxm_env/                           ← Python 虛擬環境（不用動）
├── voxelmorph_quick_test.py           ← 快速驗證安裝的腳本
└── VoxelMorph_PyTorch_實作指南.md     ← 就是這份文件
```

**atlas.npz 說明：**
- `vol`：160×192×224 的標準腦影像（float，已歸一化）
- `seg`：同尺寸的分割標籤（使用 **FreeSurfer 解剖 ID**，如 2, 3, 4, 7, 8, 10...）
- `train_avg`：訓練集平均影像（可選用）

**labels.npz 說明：**
- 包含 30 個評估用標籤 ID：`[2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,31,41,42,43,46,47,49,50,51,52,53,54,60,63]`
- 這些是 FreeSurfer 標準 ID，對應皮質、皮質下結構、腦室等

---

## 3. 環境安裝

### 3.1 系統需求

- Python 3.10.1
- CUDA GPU（訓練 3D MRI 用，建議 ≥12GB VRAM）
- virtualenv

### 3.2 建立虛擬環境

```bash
# 建立虛擬環境（用 python3.10）
virtualenv -p python3.10 vxm_env

# 啟動虛擬環境
source vxm_env/bin/activate     # Linux / macOS
# vxm_env\Scripts\activate      # Windows

# 確認版本
python --version    # Python 3.10.x
```

### 3.3 安裝套件

```bash
# 安裝 PyTorch（CUDA 13.0 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 安裝 VoxelMorph 原始碼（以可編輯模式安裝，方便修改）
pip install -e voxelmorph-code/

# 安裝其他依賴
pip install nibabel matplotlib tqdm
```

### 3.4 設定 PyTorch 後端（每次使用前必做）

VoxelMorph 預設使用 TensorFlow，要切換成 PyTorch 需要在 import 前設定：

```python
import os
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
```

或者在終端機執行：
```bash
export VXM_BACKEND=pytorch
```

### 3.5 快速驗證安裝

```bash
source vxm_env/bin/activate
python voxelmorph_quick_test.py
```

正常輸出會顯示預訓練模型的 Dice ≈ 0.6565。

---

## 4. 資料集準備（OASIS）

### 4.1 下載 neurite-oasis.v1.0

資料集官方頁面：https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/

或是https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md

下載後解壓到 `oasis/neurite-oasis.v1.0/`。

**資料集結構：**
```
oasis/neurite-oasis.v1.0/
├── subjects.txt                        ← 414 個受試者 ID 清單
├── OASIS_OAS1_0001_MR1/
│   ├── aligned_norm.nii.gz             ← 預處理後的腦影像（160×192×224，已歸一化）
│   └── aligned_seg35.nii.gz            ← 分割標籤（35 個解剖結構）
├── OASIS_OAS1_0002_MR1/
│   └── ...
└── ...（共 414 個受試者資料夾）
```

所有影像已經過 FreeSurfer 預處理（顱骨去除、仿射配準到 Talairach 空間），可直接使用。

### 4.2 轉換資料格式

VoxelMorph 訓練使用 `.npz` 格式，需要轉換：

```bash
source vxm_env/bin/activate
python oasis/prepare_oasis.py
```

腳本會自動：
- 把 414 個受試者的 `.nii.gz` 轉成 `.npz`（含 `vol` 和 `seg`）
- 隨機切分 90% train（374 筆）/ 10% test（40 筆）
- 輸出到 `oasis/oasis_npz/train/` 和 `oasis/oasis_npz/test/`

### 4.3 ⚠️ 重要：seg35 標籤對應問題

**這是最容易踩的坑！**

neurite-oasis 的 `aligned_seg35.nii.gz` 使用的是**連續編號（1~35）**，
而 `voxelmorph-code/data/atlas.npz` 的 `seg` 使用的是 **FreeSurfer 解剖 ID（2, 3, 4, 7, 8, 10...）**。

兩者不對應，如果直接用 seg35 計算 Dice 會得到幾乎為 0 的錯誤結果。

**對應關係（seg35 連續編號 → FreeSurfer ID）：**

| seg35 | FreeSurfer ID | 解剖結構 |
|-------|--------------|---------|
| 1 | 2 | Left-Cerebral-White-Matter |
| 2 | 3 | Left-Cerebral-Cortex |
| 3 | 4 | Left-Lateral-Ventricle |
| 7 | 10 | Left-Thalamus |
| 14 | 17 | Left-Hippocampus |
| 20 | 41 | Right-Cerebral-White-Matter |
| 30 | 53 | Right-Hippocampus |
| ... | ... | ... |

`test.py` 已內建完整的對應表（`SEG35_TO_FS`）並自動轉換，不需要手動處理。

---

## 5. 訓練

### 5.1 Scan-to-Atlas 訓練

每張影像都配準到同一個標準腦（`atlas.npz`）。這是最常見的腦部 MRI 配準訓練方式。

```bash
source vxm_env/bin/activate

python voxelmorph-code/scripts/torch/train.py \
    oasis/oasis_npz/train \
    --atlas voxelmorph-code/data/atlas.npz \
    --model-dir models/ \
    --gpu 0 \
    --epochs 10 \
    --steps-per-epoch 100 \
    --batch-size 1 \
    --lr 1e-4 \
    --image-loss ncc \
    --lambda 1.0 \
    --int-steps 7 \
    --int-downsize 2
```

**常用參數說明：**

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `--epochs` | 訓練幾個 epoch | 初次測試用 10，正式訓練用 1500 |
| `--steps-per-epoch` | 每個 epoch 跑幾個 batch | 100（約等於 100 張影像 / epoch） |
| `--batch-size` | 每次輸入幾張 | 1（3D 影像記憶體很大，通常只能用 1）|
| `--image-loss` | 影像相似度損失 | `ncc`（對亮度差異更魯棒，推薦） |
| `--lambda` | 形變正則化權重 | 1.0（越大形變越平滑） |
| `--int-steps` | 積分步數（微分同胚） | 7 |
| `--gpu` | 使用第幾張 GPU | 0（單卡用 0） |

訓練每個 epoch 後會自動儲存模型到 `models/XXXX.pt`（如 `models/0001.pt`）。

### 5.2 訓練 log 範例

正常訓練的 loss 應該持續下降：

```
Epoch 0001: ncc=-0.091, reg=0.042, loss=-0.049
Epoch 0002: ncc=-0.098, reg=0.039, loss=-0.059
Epoch 0003: ncc=-0.105, reg=0.037, loss=-0.068
Epoch 0004: ncc=-0.110, reg=0.036, loss=-0.074
...
Epoch 0010: ncc=-0.116, reg=0.035, loss=-0.081
```

NCC loss 是負值（因為實作是 -NCC），越負表示影像越相似。

### 5.3 訓練時間參考（GPU）

| epochs | steps/epoch | 大約時間 |
|--------|-------------|---------|
| 10 | 100 | ~20 分鐘 |
| 100 | 100 | ~3 小時 |
| 1500 | 100 | ~1.5 天 |

---

## 6. 測試與評估

### 6.1 執行測試

— 跑測試，產生 Table I + CSV

```bash
source vxm_env/bin/activate

python voxelmorph-code/scripts/torch/test.py --model models/0010.pt
```

腳本會對 `oasis/oasis_npz/test/` 裡所有 40 張影像做配準，並計算每張的 Dice 。

### 6.2 輸出範例

```
===================================================
  VoxelMorph 測試（Dice 評估）
===================================================
  模型   : models/0004.pt
  裝置   : cuda

  Atlas seg labels 數量 : 35
  評估用 labels 數量    : 30

  測試影像數量 : 40

  Subject                         Mean Dice
  ------------------------------ ----------
  OASIS_OAS1_0042_MR1               0.6234
  OASIS_OAS1_0078_MR1               0.6519
  ...

===================================================
  平均 Dice : 0.6312 ± 0.0421
  最高 Dice : 0.6731
  最低 Dice : 0.5489
===================================================
```

### 6.3 Dice 分數怎麼解讀

| Dice 範圍 | 代表意義 |
|-----------|---------|
| 0.0 ~ 0.1 | 幾乎沒有配準（通常是 bug，例如標籤對應錯誤） |
| 0.5 ~ 0.6 | 基準線（未配準，直接比較原始影像） |
| 0.6 ~ 0.7 | 初步訓練的結果（幾個 epoch） |
| 0.7 ~ 0.8 | 訓練較充分的結果 |
| > 0.8 | 相當好的配準 |

**我們的結果：**
- 未配準基準線：約 0.6565（`voxelmorph_quick_test.py` 輸出）
- 4 epochs 後：約 0.61 ~ 0.67（平均約 0.63）
- 10 epochs 後：持續改善中

### 6.4 Jacobian 行列式（進階評估）

衡量形變場的拓撲品質，負值代表 folding（形變場自我交叉，物理上不合理）：

```python
import voxelmorph as vxm
jac = vxm.py.utils.jacobian_determinant(warp_np)  # warp_np shape: (3, D, H, W)
neg_ratio = (jac < 0).sum() / jac.size
print(f'Negative Jacobian: {neg_ratio:.4%}')  # 微分同胚模型應接近 0%
```

> [!NOTE]
>
> **Table I** → Dice 均值±標準差、推論時間、負 Jacobian 數量和比例

---

## 7. 視覺化配準結果

### 7.1 執行視覺化

```bash
source vxm_env/bin/activate

python draw-img/visualize_registration.py --model models/0010.pt --csv models/dice_0010.csv
```

### 7.2 圖片說明

| 檔案                                 | 對應論文 | 內容                                                         |
| ------------------------------------ | -------- | ------------------------------------------------------------ |
| `overview_OASIS_OAS1_0047_MR1.png  ` | x        | 差值熱力圖                                                   |
| `fig4_registration_0010.png`         | Fig.4    | Moving / Fixed / Warped + 分割邊界（藍=腦室、紅=丘腦、綠=海馬） |
| `fig5_boxplot_0010.png`              | Fig.5    | 每個解剖結構的 Dice boxplot，L/R 合併平均，依分數排序        |
| `fig6_deformation_0010.png`          | Fig.6    | 彩色位移場 + 格線形變圖                                      |

---

## 8. 模型原理y

### 8.1 整體流程

```
Moving Image ──┐
               ├──> [Channel-wise Concat] ──> [U-Net] ──> [Velocity Field v]
Fixed Image  ──┘                                                   │
                                                   [Scaling & Squaring 積分]
                                                                   │
                                                         [Deformation Field φ]
                                                                   │
                              Moving Image ──> [Spatial Transformer] ──> Warped Image
```

### 8.2 各元件功能

**U-Net（Encoder-Decoder）：**
- 輸入：Moving + Fixed 兩張影像疊在一起（2 個 channel）
- 輸出：3D 速度場（3 個 channel，對應 x, y, z 方向）
- 預設架構：
  - Encoder：`[16, 32, 32, 32]`（4 層，每層 stride=2 卷積，解析度減半）
  - Decoder：`[32, 32, 32, 32, 32, 16, 16]`（4 層上採樣 + 3 層全解析度卷積）

**VecInt（Scaling-and-Squaring 積分）：**
- 速度場 → 微分同胚形變場
- 保證形變是可逆的、拓撲保持的（不會撕裂或折疊）
- `int_steps=7` 代表積分 7 次（$2^7=128$ 步細分）

**SpatialTransformer：**
- 根據形變場，對影像做空間變換（插值 warp）
- 分割標籤用 `mode='nearest'`（保持整數標籤）
- 影像用 `mode='bilinear'`（較平滑）

### 8.3 損失函數

訓練時最小化：

$$\mathcal{L} = \mathcal{L}_{sim}(I_{warped}, I_{atlas}) + \lambda \cdot \mathcal{L}_{smooth}(\phi)$$

- **$\mathcal{L}_{sim}$（NCC）**：衡量 Warped 和 Atlas 的相似程度
  - 在局部 9×9×9 窗口計算正規化互相關
  - 值域 [-1, 0]，越接近 -1 越相似（loss 越小）

- **$\mathcal{L}_{smooth}$（Grad）**：懲罰形變場的不平滑度
  - 計算形變場梯度的 L2 norm
  - 鼓勵形變平滑，防止 folding

- **$\lambda$**：平衡兩個損失的權重（`--lambda` 參數）

### 8.4 為什麼叫「非監督式」

訓練時**不需要人工標記的分割標籤**，只需要影像本身。
損失函數只用影像相似度（NCC）+ 形變平滑度（Grad）來監督。
這和傳統「給定 Ground Truth 形變場」的監督式訓練完全不同。

（若加入 Dice loss，搭配分割標籤訓練，則變成「半監督式」）

---

## 9. 常見問題與解決方案

### 問題 1：Dice 分數極低（~0.01）

**原因：** neurite-oasis 的 `aligned_seg35` 使用連續編號（1~35），但 `atlas.npz` 的 `seg` 使用 FreeSurfer 解剖 ID（2, 3, 4, 7, 8, 10...）。兩者完全不對應，導致 Dice 計算結果幾乎為零。

**解決：** `test.py` 已內建 `SEG35_TO_FS` 對應表和 `remap_seg()` 函式，執行前會自動轉換。

```python
# test.py 裡的對應表（部分）
SEG35_TO_FS = {
    0:  0,   1:  2,   2:  3,   3:  4,   5:  7,
    7: 10,  14: 17,  20: 41,  30: 53,  35: 63,
    # ...共 36 個對應（0~35）
}
```

### 問題 2：訓練指令末尾多了反引號（syntax error）

**原因：** 多行指令的最後一行不需要 `\`，若寫成 `` 7` `` 會被解析成參數值的一部分。

**解決：** 確保最後一個參數後面沒有 `\`。

### 問題 3：找不到 scripts/torch/test.py

**原因：** VoxelMorph 官方的 PyTorch 版沒有提供 `test.py`，只有 TensorFlow 版有。

**解決：** 我們自己寫了 `voxelmorph-code/scripts/torch/test.py`，就在那個位置。

### 問題 4：GPU OOM（記憶體不足）

3D 腦影像（160×192×224）佔用很多 GPU 記憶體。

**解決方案：**
- 調小 U-Net 通道數：`--enc 8 16 16 16 --dec 16 16 16 16 16 8 8`
- 增大降採樣：`--int-downsize 4`
- 確認 `--batch-size 1`（預設就是 1）

### 問題 5：能不能同時跑兩個 training？

可以，只要：
1. 各自指定不同的 `--model-dir`
2. 如果 GPU 記憶體不夠，用 `--gpu -1` 讓其中一個跑 CPU（很慢，但不衝突）

---

## 10. 參考資源

- **GitHub：** https://github.com/voxelmorph/voxelmorph
- **論文（TMI 2019）：** https://arxiv.org/abs/1809.05231
- **互動教學：** https://tutorial.voxelmorph.net/
- **neurite-oasis 資料集：** https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/
- **FreeSurfer（預處理工具）：** https://surfer.nmr.mgh.harvard.edu/
- **neurite 函式庫：** https://github.com/adalca/neurite
