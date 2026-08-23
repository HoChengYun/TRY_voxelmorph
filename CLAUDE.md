# VoxelMorph × IXI 專案交接筆記

> 給 Claude Code 的上下文文件。閱讀本文後應可直接接手任何子任務，無需重新詢問背景。
> 最後更新：**2026-08-23（第二次修訂）**

---

## 📌 文件定位：本文是唯一事實來源

專案有四份說明文件。**歷史文件的新舊順序跟直覺相反**：

| 文件 | 涵蓋範圍 | 內容停在 | 狀態 |
|------|---------|---------|------|
| **`CLAUDE.md`（本文）** | 專案總覽、IXI 主線 | 2026/08，v3 / exp8 | ✅ **唯一事實來源** |
| **`ASD/ASD相關手冊.md`** | **ASD 資料集（老師提供）那條線** | 2026/08 | ✅ **ASD 相關一律看這份** |
| `IXI/ixi相關手冊.md` | IXI 操作細節 | 2026/04，v2 / resample 時期 | 🟡 已加更正框，仍需小心 |
| `VoxelMorph_PyTorch_實作指南.md` | VoxelMorph 原理 / OASIS 時期 | 2026/03 | 🔴 最舊，多處失效 |

**該讀哪一份**：
- 要動 **ASD / FreeSurfer 標籤 / Dice** → `ASD/ASD相關手冊.md`
- 要動 **IXI 訓練或調參** → 本文（操作細節可搭 `IXI/ixi相關手冊.md`，但以本文為準）
- 想看 **VoxelMorph 原理** → `VoxelMorph_PyTorch_實作指南.md` §1、§8（其餘章節已失效）

⚠️ **本文舊版曾寫「先讀 `ixi相關手冊.md`（操作細節最完整）」——那句話是錯的**，
會把人帶去讀一份停在 v2 的文件。已刪除。兩份舊文件的具體問題見文末「舊文件的已知錯誤」。

---

## 專案概要

**任務**：用 VoxelMorph（PyTorch 版）在 **IXI** 腦部 T1 MRI 上做 **scan-to-atlas** 非剛性配準，
atlas 為 **MNI152 ICBM 2009c Asymmetric**（自製，非官方 OASIS atlas）。

**當前狀態**：前處理／訓練／評估／視覺化都已跑通，完成 8 組實驗（exp1–exp8）。
ASD（老師提供）那條線的**前處理已跑完**（167 顆 → train 150 / test 17），
訓練在**另一台機器**上跑（見「待辦 1」與 `ASD/ASD相關手冊.md`）。

⚠️ **本專案不含 TransMorph**。TransMorph 在 `D:\MyHome\MRI\TransMorph\`，有自己的 `CLAUDE.md`。
根目錄的 `TransMorph_Report.docx` 只是報告備份，與本專案程式碼無關。

---

## 執行環境

- **OS**：Windows（原生，非 WSL）
- **虛擬環境**：`vxm_env\`（venv，不進 git）
- **PyTorch**：`2.10.0+cu130`（CUDA 13.0）
- **TensorFlow**：`2.21.0` — 只有 antspynet 去顱骨會用到；**Windows 原生 TF ≥ 2.11 不支援 GPU，去顱骨走 CPU，正常現象**
- **關鍵套件**：`antspyx==0.6.3`、`antspynet==0.3.2`、`neurite==0.2`、`nibabel==5.4.2`、`numpy==2.2.6`
- **GPU 檢查**：`python cuda_cheak.py`

```powershell
cd C:\Users\h4524\claude_cheng
.\vxm_env\Scripts\activate
```

⚠️ `requirements.txt` 裡**沒有 torch**（CUDA wheel 另外裝的）。重建環境時要另外裝對應 CUDA 版本的 PyTorch。

---

## 目錄結構

```
C:\Users\h4524\claude_cheng\
├── voxelmorph-code\                    # VoxelMorph 官方 repo（含本地修改）
│   ├── voxelmorph\                     # 套件本體（torch\losses.py、torch\networks.py 在這）
│   ├── data\
│   │   ├── atlas.npz                   # OASIS atlas：keys = vol / seg / train_avg
│   │   ├── labels.npz                  # ⭐ Dice 評估用的 30 個 FreeSurfer 標籤 ID
│   │   ├── generated_uncond_atlas.npz  prob_atlas.npz
│   │   ├── prob_atlas_T1_stats.npz     prob_atlas_mapping.npz
│   │   └── test_scan.npz
│   └── scripts\torch\
│       ├── train.py                    # 主訓練腳本
│       ├── test_oasis.py               # OASIS 用（需要 seg，IXI 跑會報錯）
│       ├── test_ixi.py                 # ⭐ IXI 用：單一模型 NCC / SSIM → CSV
│       ├── batch_test_ixi.py           # ⭐ 掃全部 .pt，畫 epoch 曲線
│       ├── batch_test_oasis.py         register.py
│       └── train\train_NCCPatchSize.py # 可調 --ncc-win 的訓練變體（⚠️ 目前尚未用它跑過任何實驗）
├── ASD\                                # ⭐ ASD 資料集（老師提供）＋ FreeSurfer 標籤接入
│   ├── ASD相關手冊.md                  # ⭐ ASD 這條線的完整操作手冊，先讀這個
│   ├── preprocess_fs.py                # FreeSurfer 產物 → npz（含 seg）
│   ├── verify_seg_transform.py         # 驗證 affine 共用 + 最近鄰內插（已實測 6/6 通過）
│   ├── verify_one_subject.py           # 單顆量化驗證（含左右翻轉檢查）
│   ├── subjects_final.txt              # ✅ FINAL 清單（167 個 ID）
│   ├── groups.txt                      # 歸戶對照（目前不需要，見手冊 §3）
│   ├── ASD_data\norm\  ASD_data\aseg\  # ✅ 已落地：各 167 個 .nii.gz，共 285 MB
│   ├── fs_check\                       # --only 單顆驗證輸出
│   ├── run_preprocess.py               # 前處理包裝（檢查→預覽→執行→抽驗）
│   ├── run_train.py                    # 訓練包裝（--check-only / --resume）
│   └── ASD_preprocessed_v1\            # ✅ train 150 / test 17 + split.json（1.28 GB）
├── IXI\
│   ├── IXI-T1\                         # 原始 IXI T1（581 張 .nii.gz）
│   ├── mni_icbm152_nlin_asym_09c_nifti\ # 下載的 MNI152 2009c
│   ├── make_atlas.py                   # 製作 atlas（→ .npz + .nii.gz）
│   ├── preprocess_ixi.py               # ⭐ 前處理主腳本
│   ├── visualize_preprocess_ixi.py     SeeHeader.m   read_nii_header.py
│   ├── atlas_mni152_09c_v2.{npz,nii.gz}  # resample 版
│   ├── atlas_mni152_09c_v3.{npz,nii.gz}  # crop 版（現行）
│   ├── IXI_preprocessed\               # v1（舊，🔴 已不可重現）train 522 / test 59
│   ├── IXI_preprocessed_v2\            # v2       train 522 / test 59
│   ├── IXI_preprocessed_v3\            # v3（現行）train 522 / test 59 + nii\
│   ├── preprocess_vis\                 # --vis 模式輸出
│   ├── verify\                         # 方向 / 前處理驗證腳本 ×3
│   ├── orientation_verify\             NPZtoNII\   orientation_check.png
├── draw-img\
│   ├── visualize_reg_ixi.py            # ⭐ 學術級視覺化 + 存 nii
│   ├── visualize_reg_oasis.py          plot_epoch_curve.py
├── models\                             # 所有訓練權重（.gitignore，不進 git）
│   ├── exp1\  exp2_IXI\  exp3_IXI\  exp4\ … exp8\
│   ├── atlas_creation_uncond_NCC_1500.h5   # 官方 TF 版預訓練權重
│   └── vxm_dense_brain_T1_3D_mse.h5        # 官方 TF 版預訓練權重
├── share_models\                       # 空資料夾，用途不明
├── log\                                # 訓練 stdout + 當初的指令
├── oasis\                              # OASIS 前處理（舊線，目前不動）
├── meeting報告\                        # 簡報 pptx（.gitignore）
├── 前一AI擔心的\                        # 文件稽核報告（另一個 session 產出）
├── FreeSurfer_到_VoxelMorph_交接.md    # FreeSurfer 端寫的接入說明
├── VoxelMorph_PyTorch_實作指南.md      # 🔴 最舊，見文末
└── IXI\ixi相關手冊.md                  # 🟡 停在 v2，見文末
```

---

## 資料流水線

```
IXI-T1 原始 .nii.gz (581)
  ↓ make_atlas.py：MNI152 2009c → 去顱骨 → clip 1~99% → [0,1] → 調整成 (192,224,192)
  ↓                輸出 atlas .nii.gz（帶 header，給 ANTs）+ .npz（給 train.py）
  ↓ preprocess_ixi.py
  │   ① ants.n4_bias_field_correction()
  │   ② antspynet.brain_extraction()   ← CPU，最慢的一步
  │   ③ ants.registration(type_of_transform='Affine') 對到 atlas
  │   ④ shape 驗證（必須等於 atlas）
  │   ⑤ clip 1~99% percentile → min-max 正規化到 [0,1]
  │   ⑥ np.savez_compressed(vol=img)
  ↓ train/ (522, 90%) + test/ (59, 10%)   seed=42
  ↓ train.py（scan-to-atlas，非監督）
  ↓ batch_test_ixi.py → 挑最佳 epoch
  ↓ visualize_reg_ixi.py → 出圖 + 存 nii
```

**npz 格式**：key 只有 `vol`，shape `(192, 224, 192)`，`float32`，值域 `[0, 1]`。
**目前沒有 `seg`**，所以評估只能用 NCC / SSIM，**報不出 Dice**。

---

## ⚠️ 資料版本：v1 / v2 / v3（最容易搞混的地方）

三個版本的 shape 都是 `(192,224,192)`、key 都只有 `vol`，**從檔案看不出差別**，
差別在 **atlas 是怎麼從 MNI152 原始 (193,229,193) 變成 (192,224,192) 的**：

| 版本 | 日期 | `make_atlas.py --method` | spacing | 說明 |
|------|------|--------------------------|---------|------|
| v1 (`IXI_preprocessed`) | 2026-03 | 早期 `scipy.ndimage.zoom` | 非 1mm | 舊版，🔴 **atlas 檔案已刪除，不可重現** |
| v2 (`IXI_preprocessed_v2`) | 2026-04-27 | `resample` | ≈(1.005, 1.022, 1.005) mm | **exp4–exp8 全部用這版** |
| **v3 (`IXI_preprocessed_v3`)** | 2026-05-19 | **`crop`** | **精確 1mm** | **現行**，裁掉背景而非縮放 |

> `--method crop` 直接裁掉邊緣背景 voxel，**spacing 維持精確 1mm**，
> 不像 `resample` 會把體素稍微拉長。新實驗一律用 v3 + `atlas_mni152_09c_v3.npz`。

🔴 **鐵則：訓練用的 `--atlas` 版本必須跟 `datadir` 的資料版本一致。**
混用會出現 `Sizes of tensors must match` 或（更糟）安靜地訓練出對不準的模型。

⚠️ **v1 已不可重現**：`IXI/atlas_mni152_09c_v1.*` 不存在（只剩 v2 / v3），
且當初用的是早期 `scipy.ndimage.zoom`。`IXI_preprocessed/` 那批資料留著也重跑不出來。

---

## 🔬 超參數：λ 的尺度取決於 image-loss（最容易誤判的一點）

**同一個 λ 在 MSE 和 NCC 之間意義完全不同**，因為兩者損失量級差兩個數量級。

論文（TMI 2019）Fig. 7 分兩張圖，**x 軸刻度不一樣**：

| image-loss | 損失量級 | 論文最佳 λ | `train.py` 預設 λ |
|---|---|---|---|
| MSE | ~0.005（影像已正規化到 [0,1]）| **0.01 – 0.02** | 0.01 ✅ 落在最佳區 |
| NCC | ~1（`-mean(cc)`）| **≈ 1 – 2** | 0.01 ❌ 小了兩個數量級 |

用 `log/exp*.txt` 最後一步的損失拆項實測（`train.py` 會印 `loss: 總計 (影像項, 平滑項)`）：

| 實驗 | image-loss | λ | 影像項 | 平滑項 | **平滑佔比** |
|------|-----------|---|--------|--------|------------|
| exp4 | ncc | 0.005 | -0.238 | 0.00282 | 1.17% |
| exp5 | ncc | 0.05 | -0.237 | 0.00799 | 3.26% |
| exp7 | ncc | 0.01 | -0.234 | 0.00301 | 1.27% |
| exp8 | ncc | 0.01 | -0.252 | 0.00201 | 0.79% |
| **exp6** | **mse** | **0.01** | 0.00204 | 0.00066 | **24.53%** |

🔴 **所有 NCC 實驗的平滑項只佔損失 1–3%，形同沒有正則化。**
exp5 把 λ 調大 10 倍也只從 1.2% 變 3.3%。**要用 NCC 的話，λ 應該往 0.5–2 試，不是在 0.05 以下打轉。**

📌 **最舊的 `VoxelMorph_PyTorch_實作指南.md` §6 其實寫對了**：它的範例指令是
`--image-loss ncc --lambda 1.0`，正好命中論文對 CC 的最佳值。
**是後續 exp3–exp8 偏離了這個設定**（改成 0.005 / 0.01 / 0.05），不是文件寫錯。

⚠️ 誠實邊界：論文主模型是非微分同胚（`int_steps=0`），repo 預設多了積分與 `loss_mult`，
所以 λ≈1 不保證能原封不動搬過來。但「差兩個數量級」這點，論文圖與 log 實測互相印證。

---

## 常用指令

全部在 `C:\Users\h4524\claude_cheng` 底下、`vxm_env` 啟動後執行。

### 製作 atlas
```powershell
python IXI\make_atlas.py `
    --t1   IXI\mni_icbm152_nlin_asym_09c_nifti\mni_icbm152_t1_tal_nlin_asym_09c.nii `
    --mask IXI\mni_icbm152_nlin_asym_09c_nifti\mni_icbm152_t1_tal_nlin_asym_09c_mask.nii `
    --target-shape 192,224,192 `
    --method crop `
    --out IXI\atlas_mni152_09c_v3
```

### 前處理（全批，數小時；`--skip-done` 預設開啟可續跑）
```powershell
python IXI\preprocess_ixi.py `
    --out-dir IXI\IXI_preprocessed_v3 `
    --atlas   IXI\atlas_mni152_09c_v3.nii.gz `
    --save-nii
```

⚠️ `preprocess_ixi.py` 的 `--target-shape` **已移除**，輸出 shape 直接由 `--atlas` 決定。
（`make_atlas.py` 的 `--target-shape` **仍然存在**，兩者不要混淆。）

### 訓練
```powershell
python voxelmorph-code\scripts\torch\train.py IXI\IXI_preprocessed_v3\train `
    --atlas IXI\atlas_mni152_09c_v3.npz `
    --model-dir models\expN `
    --epochs 500 --gpu 0 `
    --lambda 0.005 --image-loss ncc > .\log\expN.txt 2>&1
```

🔴 **`--image-loss` 預設是 `mse`，不是 ncc。** 想跑 NCC 一定要明寫（exp6 就是沒寫，跑成 MSE）。
其他預設值：`--lr 1e-4`、`--lambda 0.01`、`--steps-per-epoch 100`、`--batch-size 1`、
`--int-steps 7`、`--int-downsize 2`。

⚠️ `datadir` 是**位置參數**，不是 `--datadir`。

⚠️ **`train.py` 沒有 `--ncc-win`**，NCC 窗格永遠是 `9`
（`voxelmorph/torch/losses.py:26`：`win = [9] * ndims if self.win is None`）。
要調 patch size 必須換腳本：
```powershell
python voxelmorph-code\scripts\torch\train\train_NCCPatchSize.py IXI\IXI_preprocessed_v3\train `
    --atlas IXI\atlas_mni152_09c_v3.npz --model-dir models\expN `
    --epochs 250 --gpu 0 --image-loss ncc --ncc-win 5
```

### 評估
```powershell
# 掃全部 epoch，畫四指標曲線（先做這步挑 epoch）
python voxelmorph-code\scripts\torch\batch_test_ixi.py `
    --model-dir models\expN --atlas IXI\atlas_mni152_09c_v3.npz `
    --test-dir IXI\IXI_preprocessed_v3\test --out-dir models\expN --step 1 --gpu 0

# 單一模型詳細評估 → CSV
python voxelmorph-code\scripts\torch\test_ixi.py `
    --model models\expN\0155.pt --atlas IXI\atlas_mni152_09c_v3.npz `
    --test-dir IXI\IXI_preprocessed_v3\test --gpu 0
```

### 視覺化（挑好 epoch 後）
```powershell
python draw-img\visualize_reg_ixi.py `
    --model models\expN\0155.pt --atlas IXI\atlas_mni152_09c_v3.npz `
    --test-dir IXI\IXI_preprocessed_v3\test `
    --out-dir models\expN\0155 --save-nii --gpu 0
```
輸出 5 種圖：triplanar / checkerboard / warped grid / overlay / Jacobian map，
加 `--save-nii` 另存 `warped_*.nii.gz` + `warp_*.nii.gz`。

---

## 評估指標

| 指標 | 方向 | 說明 |
|------|------|------|
| **NCC** | 越高越好 | ⚠️ 見下方警告，目前已飽和 |
| **SSIM** | 越高越好 | 結構相似度，比 NCC 接近人眼感知 |
| **%\|J\|≤0** | 越低越好 | Jacobian 非正比例＝形變場**折疊**。⚠️ 判準見下 |
| **Smoothness** | 適中 | 位移場梯度能量。太低＝幾乎沒形變；太高＝形變過激 |
| ~~Dice~~ | — | 🔴 **目前做不到**，npz 沒有 `seg`。見「待辦 1」 |

### ⚠️ 報表的 NCC 是會飽和的弱指標

`test_ixi.py:61` 的 `ncc()` 是**整顆 volume 的全域 Pearson 相關，含背景**。
去顱骨又對齊 atlas 之後，背景是大片 0 對 0，貢獻大量「完美相關」。

這跟**訓練損失用的局部窗格 CC²**（論文 Eq. 6，n=9）不是同一個東西
——所以會看到「loss 的 NCC ≈ 0.24，報表的 NCC ≈ 0.99」，兩者不能互相對照。

🔴 **後果：exp4–exp8 的 NCC 全擠在 0.986–0.992，指標已飽和，區分不出模型好壞。**
這正是要接 Dice 的理由。

### ⚠️ %|J|≤0 的合理標準（論文 Table I 實測值）

| 方法 | Dice | %\|J_φ\|≤0 |
|---|---|---|
| ANTs SyN (CC) | 0.749 | 0.185% |
| NiftyReg (CC) | 0.755 | 0.793% |
| **VoxelMorph (CC)** | 0.753 | **0.366%** |
| VoxelMorph (MSE) | 0.752 | 0.184% |

論文原文：所有方法都會出現這種小島，但在 **99.4–99.9%** 的 voxel 上仍是微分同胚。

🔴 **本文舊版把 0.10% 標成警訊，那是誤判——0.1% 比論文發表的模型還好。**
真正該避開的是 exp4 ep477 那種 **2.26%**。
反過來說，ep69 的 **0.003% 低到反常**，比較像形變量太小（欠擬合），不見得是好事
——這點沒有證據下定論，但「NCC 高 + %|J|≤0 越低越好」這個判準的來源不明，別當鐵律。

---

## 實驗記錄

⚠️ **exp4–exp8 全部使用 v2 資料 + v2 atlas**（已從 `log/exp*_script.txt` 逐一確認）。

| 實驗 | 資料 | atlas | epochs | image-loss | λ | NCC win | int-steps | int-downsize | 最佳 epoch |
|------|------|-------|--------|-----------|---|---------|-----------|--------------|-----------|
| exp1 | OASIS | OASIS | 405 存檔 | — | — | — | — | — | — |
| exp2_IXI | v1 | v1 | 105 存檔 | — | — | — | — | — | — |
| exp3_IXI | v1/v2? | — | 300 | ncc | — | 9 | 7 | 2 | 0295 |
| **exp4** | v2 | v2 | 500 | ncc | **0.005** | 9 | 7 | 2 | **0069** |
| exp5 | v2 | v2 | 500 | ncc | **0.05** | 9 | 7 | 2 | 0065 |
| **exp6** | v2 | v2 | 250 | **mse**（預設）| **0.01**（預設）| — | 7 | 2 | 0153 |
| exp7 | v2 | v2 | 250 | ncc | **0.01**（預設）| **9**（預設）| 7 | 2 | 0155 |
| exp8 | v2 | v2 | 250 | ncc | **0.01**（預設）| 9 | **3** | **1** | 0122 |

**ep250/500 的最終指標**：exp5 NCC .988 / SSIM .959 / Jneg .0006%；exp6 .987 / .939 / 3e-5%；
exp7 .989 / .956 / .006%；exp8 .992 / .972 / **.10%**（.10% 屬正常範圍，見上）。

### 🔴 舊版本文對 exp7 / exp8 的描述是錯的

舊版寫「exp7 = NCC patch size 5」「exp8 = NCC patch size 實驗」，**兩者都不對**。
從 `log/exp7_script.txt`、`log/exp8_script.txt`（UTF-8）解出的實際指令：

```
exp7: train.py ... --epochs 250 --gpu 0 --image-loss ncc
exp8: train.py ... --epochs 250 --gpu 0 --image-loss ncc --int-steps 3 --int-downsize 1
```

再從 `.pt` 存的 config 獨立確認（`torch.load(...)['config']`）：exp8 是 `int_steps=3, int_downsize=1`，
其餘全是 `7 / 2`。兩邊一致。

- **exp7** 用的是 `train.py`（**不是** `train_NCCPatchSize.py`），λ 與 win 全走預設 → λ=0.01、win=9
- **exp8** 是**積分參數實驗**，與 patch size 無關

👉 **`train_NCCPatchSize.py` 至今一次都沒用來跑過實驗。** git commit message
「test different ncc patch size」同樣是誤記。

### ⚠️ exp8 一次動了三個變因

`train.py:143` 是 `Grad('l2', loss_mult=args.int_downsize)`。
`--int-downsize 1` 讓 `loss_mult` 從 2 變 1，**等於把平滑懲罰的實際權重砍半**。
所以 exp8 同時改了：積分步數（7→3）、形變場解析度（半解析度→全解析度）、正則化強度（砍半）。
**不是單一變因實驗，結果無法歸因。**

### 📌 exp6 其實是「作者預設跑法」

`log/exp6_script.txt` 的指令一個超參數都沒改，等於 `train.py` 的完整預設值。
舊版本文把它記成「漏寫 `--image-loss ncc` 的失誤」，但從論文角度看，
**exp6 是唯一一個 λ 落在論文建議區間內的實驗**（MSE 最佳 λ=0.01–0.02），
平滑佔比 24.5% 也是唯一像樣的正則化強度。

⚠️ 一個落差：repo 預設 `int_steps=7` 是**微分同胚版本**（速度場參數化，Dalca et al.），
而 TMI 2019 論文 Table I 的主結果是**非微分同胚的位移場版本**（`int_steps=0`）。
「跑作者預設」與「複現論文表格」嚴格說不是同一件事。

### exp4 最佳 epoch 選擇的細節（來自 `ixi相關手冊.md` §9.2）

| 指標 | ep 69（當初選這個）| ep 477（NCC 最高）| ep 494（SSIM 最高）|
|------|---------------|------------------|-------------------|
| NCC | 0.9863 | 0.9905 | 0.9901 |
| SSIM | 0.9455 | — | 0.9645 |
| %\|J\|≤0 | 0.003% | ~2.26% | ~2.26% |

---

## `log/` 實際內容

⚠️ **舊版本文說「exp3–exp6 的指令在 `log/exp*_script.txt`」不準確。** 實際：

```
exp3.txt              ← 只有 stdout，沒有 script log
exp4.txt   exp4_script.txt.txt   ← ⚠️ 副檔名重複，尚未改名
exp5.txt   exp5_script.txt
exp6.txt   exp6_script.txt
exp7.txt   exp7_script.txt       ← 2026-08-23 補上
exp8.txt   exp8_script.txt       ← 2026-08-23 補上
train_IXI.txt   train_oasis.log
```

- **exp3 沒有 script log**，參數仍不明。
- 有 script log 的是 **exp4–exp8**。
- 👉 之後每跑一個實驗，**指令一定要存成 `log/expN_script.txt`**。

🔴 **編碼陷阱**：`log/*.txt` **混用兩種編碼**——
`exp3–exp6` 的 stdout 是 **UTF-16LE**，但 `exp*_script.txt` 是 **UTF-8**。
不要假設全部是 UTF-16，用下面這種偵測式讀法：

```python
raw = open(path, 'rb').read()
for enc in ('utf-16', 'utf-8', 'cp950'):
    try:
        t = raw.decode(enc)
        if '\ufffd' not in t: break
    except Exception: pass
```

---

## 已知問題 / 踩過的坑

| 問題 | 原因 | 解法 |
|------|------|------|
| `Sizes of tensors must match … Expected 192 but got 193` | atlas npz 與訓練資料版本不一致 | 確認 `--atlas` 跟 `datadir` 是同一版 |
| 訓練跑出來 loss 是正的、很小 | 忘了 `--image-loss ncc`，跑成 MSE | 明寫 `--image-loss ncc`（NCC loss 是負值） |
| NCC 訓練出來會折疊 / 形變過激 | λ 對 NCC 而言小了兩個數量級 | 見「超參數」節，λ 試 0.5–2 |
| `RuntimeError: size XXX not divisible` | 影像維度不能被 16 整除（U-Net 4 層下採樣） | **在 `make_atlas.py` 指定 `--target-shape`**（不是 preprocess_ixi.py，它已移除該旗標） |
| `CUDA out of memory` | 192×224×192 在 8GB GPU 上很緊 | `--batch-size 1`，或改用更小的 target shape |
| 視覺化 / 推論卡住很久 | 沒加 `--gpu 0`，走 CPU 跑 3D U-Net | 一律加 `--gpu 0` |
| `test_oasis.py` 讀 `atlas['seg']` 報 KeyError | IXI atlas 沒有 seg | 改用 `test_ixi.py` |
| NCC loss 報 CUDA 錯誤 | `losses.py:28` 的 `sum_filt` 寫死 `.to("cuda")` | NCC 訓練必須有 GPU |
| TF Keras 版本衝突 | neurite 預設走 TF backend | 腳本最前面加 `os.environ['NEURITE_BACKEND'] = 'pytorch'` |
| 模型在 CPU 上跑 | 忘了搬到 GPU | `model.to(device)` 要放在 `model.eval()` 之前 |
| antspynet TF GPU warning（Windows）| TF ≥2.11 原生 Windows 無 GPU | 可忽略，去顱骨走 CPU 不影響結果 |
| `tf.function retracing` warning | 各張影像 shape 略有不同 | 可忽略 |
| `--skip-done` 想關掉 | 它預設就是 True | 加 `--no-skip-done` 或刪掉輸出資料夾重跑 |
| 灰白質界面略模糊 | NCC 是 patch-based，對高頻細節不敏感 | 縮小 NCC win（**尚未實驗過**）；或先修正 λ |

### 🔧 給 Claude 的操作注意事項

- **`log/` 的編碼混用兩種**，見上一節。`requirements.txt` 是 UTF-16LE。
- **`models/`、`IXI/`、`draw-img/`、`oasis/`、`meeting報告/` 都在 `.gitignore` 裡**（只保留 `*.py`/`*.md`）。
- `models/exp1` 有 201 個 `.pt`、`exp4` 有 501 個——**不要遍歷讀取**，只讀 `epoch_curve.csv`。
  （exp5–exp8 已清理，各只留最佳 epoch 的 `.pt` + `epoch_curve.csv/png`。）
- 這是 **Windows 原生環境**，路徑用 `\`，指令用 PowerShell 語法（換行用反引號 `` ` ``）。
- **動 git 之前先問使用者。** 目前 untracked：
  `CLAUDE.md`、`FreeSurfer_到_VoxelMorph_交接.md`、`ASD/`、`log/exp7*`、`log/exp8*`、`前一AI擔心的/`，
  以及已修改的 `IXI/ixi相關手冊.md`、`VoxelMorph_PyTorch_實作指南.md`、`.gitignore`。
  ⚠️ 這幾份是目前最重要的資產，卻都還沒進版控。

---

## 待辦

### 1. 🟢 接入 FreeSurfer 標籤（前處理已完成）→ 🔴 Dice 評估仍待寫

> 🔴 **程式已經寫好了，不要重寫。**
> **完整操作細節在 `ASD/ASD相關手冊.md`**，先讀那份；程式是 `ASD/` 底下這兩支。

| 檔案 | 狀態 |
|---|---|
| `ASD/preprocess_fs.py` | ✅ 已完成 |
| `ASD/verify_seg_transform.py` | ✅ 已完成**並實測通過 6/6** |

`preprocess_fs.py` 已實作：
- 不做 N4、不做去顱骨（FreeSurfer 的 `nu.mgz` / `brain.mgz` 已處理過），保留 `--n4` / `--brain-extract` 可強制開啟
- aseg 用**與影像完全相同**的 Affine 變換 + 最近鄰內插搬過去，搬完若出現非整數值直接 raise
- **切分以「受試者」為單位**，歸戶規則做成外部設定檔 `--group-map`（TSV）
- 內建排除清單 `['A043', 'T085', 'A012', 'T065']`
- `--dry-run`（只看歸戶與切分）、`--only <subject>`（先驗 1 顆）、輸出 `split.json`
- 🔒 **執行閘門**：沒有 `--list-is-final` 時**拒絕批次執行**（exit code 2）

**已驗證的事實**（`verify_seg_transform.py`，ANTsPy 0.6.3）：
用同一個 `reg['fwdtransforms']` 把影像與標籤都以 nearestNeighbor 搬過去，
再把搬完的影像重新量化、與搬完的標籤逐 voxel 比對 → **0 / 8,257,536 不一致**。
反證：改用 `linear` 會捏造出 1,536,912 個原標籤集裡沒有的值。

**訓練端相容性也已驗證**：npz 有 `vol` + `seg` 兩個 key 時，
`voxelmorph/py/utils.py:63` 走 `npz[np_var]`（`np_var` 預設 `'vol'`），會正確取到 `vol` 不會拿到 `seg`。
`train.py` 用預設值實跑 2 epoch 正常。**VoxelMorph 那邊不用改。**

**✅ 已解決的**：
- **混掃描全面檢查**：2026-08-23 完成，167 個資料夾、**異常 0**，
  167 × 192 = 32,064 與實際檔案數完全閉合。閘門條件已滿足。
- **最終清單**：167 顆，已複製一份到 `ASD/subjects_final.txt` 進版控。
- **影像來源**：使用者選 `norm.mgz`（不是 `brain.mgz`），167 顆全體一致。

**✅ 前處理已完成（2026-08-23）**：
資料已落地 `ASD/ASD_data/norm`＋`aseg`（各 167 個，285 MB），
批次前處理跑完 **0 失敗、0 個標籤消失**，輸出 `ASD/ASD_preprocessed_v1/`
（train 150 / test 17，**1.28 GB**）。

**還卡著的一件事**：
🟠 **A013 / A0131 / A0132、A016_1 / A016_2 是否同一人** —— 要問老師。
已用程式掃過全部 167 個 ID，**這種命名曖昧全批只有這兩組**，沒有第三處。
⚠️ 目前採「都是不同人」的**暫定假設**（使用者決定，`--grouping none`），
若錯會造成 data leakage，**要發表必須在方法學說明或先確認**。
好消息是可逆：寫一份 `ASD/groups.txt` 用 `--group-map` 重跑切分即可，不用改程式。
詳見 `ASD/ASD相關手冊.md` §3。

**清單**：
```
D:\MyHome\MRI\FreeSurfer\docs\ASD_可用清單_FINAL.txt   ★ 最終版（167 個）
D:\MyHome\MRI\FreeSurfer\docs\ASD_全部資料夾清單.txt   （170 個，含壞掉的）
```
⚠️ 舊的 `ASD_可用清單_暫定.txt`（166）**已被刪除**，不要再引用。

用 FINAL 清單 + `--grouping none` 的實跑結果：
**167 個掃描 → 167 位受試者，train 150 人/150 掃描、test 17 人/17 掃描，無人橫跨。**

**⭐ Dice 評估腳本尚未撰寫。** 可直接用 `voxelmorph-code/data/labels.npz`
（已實際載入確認：30 個 FreeSurfer 標籤 ID，`int64`）：
```
[2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24,
 28, 31, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60, 63]
```
這正好對應論文 Table I 的 30 個結構（收錄標準：在所有測試受試者中體積都 ≥ 100 voxel）。

---

### 2. 🔴 修正 λ 並重跑 NCC 實驗

見「超參數」節。目前所有 NCC 實驗的正則化都形同虛設。
建議在 v3 資料上跑 λ = 0.5 / 1 / 2，與 exp6（作者預設 MSE）對照。

---

### 3. NCC patch size 系統性比較（**一次都還沒做過**）

`train_NCCPatchSize.py` 存在但從未使用。要做的話：
在**同一份資料（v3）+ 同 λ** 下跑 `--ncc-win 5 / 7 / 9`。

---

### 4. （備案）SynthSeg 產生偽標籤

若 FreeSurfer 那批資料延宕，可用 `mri_synthseg` 幾分鐘/張產生 seg 先把 Dice 流程打通。

⚠️ **標籤方法不一致本身就是 confound**：若 IXI 用 SynthSeg、ASD 用 aseg，
兩邊的 Dice **不能直接比較**（差異裡混雜了「配準品質」與「兩套分割方法」，分不開）。
規則：**在哪一批算 Dice，那批就要用同一套標籤方法。**

FreeSurfer 端建議的兩段式設計（**尚未定案，決定權在使用者與老師**）：
- 第一段：IXI 上 train + test，標籤統一用 SynthSeg → 主結果，cohort 一致
- 第二段：ASD 當 **external validation**，明講是跨 cohort / 跨機器

---

## 舊文件的已知錯誤（尚未修正）

以下問題**都還沒改**，改之前請先問使用者。來源：`前一AI擔心的/前一AI擔心的.md`（我已逐項複驗）。

### `IXI/ixi相關手冊.md`（停在 v2）

| 位置 | 問題 | 嚴重度 |
|---|---|---|
| §6.1 表格 | 把「裁切 cropping」標成 ❌「可能砍到腦組織邊緣」——但 **v3 現行做法就是 `--method crop`**，照做會退回 v2 | 🔴 最危險 |
| 全篇 | 沒有 v1/v2/v3 的概念；atlas 檔名寫 `atlas_mni152_09c.nii.gz` / `_resize.npz`，**兩個都不存在** | 🔴 |
| 全篇 | **完全沒有「atlas 版本必須與資料版本一致」這條鐵則** | 🔴 |
| §4.5 參數表、§6 常見問題 | 還在教 `preprocess_ixi.py --target-shape`，**現在會直接報 unrecognized argument** | 🔴 |
| §5.1 | 訓練指令寫 `--datadir`（實際是位置參數），且 PowerShell 區塊用 `\` 換行（該用反引號） | 🟡 |
| §9.3 | 建議「自己去 train.py 加 --ncc-win」——`train_NCCPatchSize.py` 早就存在 | 🟡 |
| §7 | 結構過舊：`verify_orientation.py` 等三支已移到 `IXI/verify/`；缺 `NPZtoNII/`、`orientation_verify/`、`preprocess_vis/` | 🟢 |
| 4 處 | `test.py` → 應為 `test_oasis.py` | 🟡 |

### `VoxelMorph_PyTorch_實作指南.md`（最舊，停在 OASIS 時期）

| 問題 | 說明 |
|---|---|
| **整份是 Linux 語法** | `source vxm_env/bin/activate` → 應為 `.\vxm_env\Scripts\activate` |
| **腳本全部改名了** | `test.py`→`test_oasis.py`/`test_ixi.py`；`batch_test.py`→`batch_test_*.py`；`visualize_registration.py`→`visualize_reg_*.py`（16 處提到 `test.py`）|
| **`train_avg` 說明錯誤** | 指南 §2 說是「訓練集平均影像」→ ❌。實測 `atlas.npz` 的 `train_avg` 是 `shape=(256,) float64`，是**各標籤的平均 Dice**（256 = FreeSurfer label ID 範圍 0–255），手冊 §1.1 的說法才對 |
| ~~`--lambda 1.0` 建議值誤導~~ | ✅ **這項不是錯的，反而是對的**。指南 §6 的指令是 `--image-loss ncc --lambda 1.0` 配套出現，正好命中論文對 CC 的最佳 λ（≈1–2）。稽核報告拿 `train.py` 預設 0.01 去比而未考慮損失類型，判斷有誤。**真正的問題是後續 exp3–exp8 偏離了這份最舊文件的正確設定** |
| **沒寫 `--image-loss` 預設是 mse** | 指南的範例有明寫 `ncc`，但沒提「不寫就會變 mse」。exp6 就是踩這個 |
| **完全沒有 IXI 這條線** | 主線早就轉到 IXI + MNI152 |
| **基準線數字矛盾** | §6.3 說未配準基準 0.6565，§6.5 說 Affine only 0.584，沒解釋差別 |
| **§4.3 `SEG35_TO_FS` 只列 7 行就 `...`** | 接 aseg 標籤時完整表會有用 |
| **Markdown 壞掉** | §6.5、§9 問題6 混用 ``` 和 `~~~` 導致巢狀壞掉 |
| **§8 標題打錯字** | 「模型原理y」多一個 y |
| **§6.5 有 AI 回話殘留** | 正文夾了一句「Created batch testing script for VoxelMorph models」 |

### 稽核報告本身的錯誤

`前一AI擔心的/前一AI擔心的.md` §2.5 說「exp7 / exp8 就是用 `train_NCCPatchSize.py` 跑的」
——**這是錯的**，已由 `log/exp*_script.txt` 與 `.pt` config 雙重推翻（見「實驗記錄」節）。
該文件 §4.7 的 git status 也已過時。

---

## 參考

- **VoxelMorph 論文**：Balakrishnan et al., *VoxelMorph: A Learning Framework for Deformable Medical Image Registration*, IEEE TMI 38(8):1788–1800, 2019（PDF 在專案根目錄）
  - Eq. (6) 局部 CC，**n = 9**（對應 `losses.py` 預設 win=9）
  - Eq. (7) 平滑項 = 位移場空間梯度的 L2（對應 `Grad('l2')`）
  - Eq. (4) `L_us = L_sim + λ·L_smooth`
  - Fig. 7 λ 敏感度；Table I 各方法的 Dice 與 %|J|≤0
  - 論文的資料前處理也是「affine + FreeSurfer 去顱骨」，標籤同樣來自 FreeSurfer
- **官方 repo**：https://github.com/voxelmorph/voxelmorph
- **IXI 資料集**：https://brain-development.org/ixi-dataset/
- **MNI152 2009c atlas**：https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/
