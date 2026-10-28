# 照原作程式重現：DWP、RCN、TransMorph、TransMorph-diff（cmd 版）

> **這份在做什麼**：新方法的第一步先把原作程式原封不動跑在我們的資料上，只改逼不得已的（2026-10-28 使用者決定，手冊 §31）。
> 之前跑過的 mix_cascade／mix_pyramid 系列、tm_mixv2、tm_diff_mixv2 都是**舊設定**（照原作的機制自己寫、訓練設定對齊我們的 VoxelMorph），保留不動。
>
> | 方法 | 原作程式 | 怎麼跑 | 逼不得已改的 |
> |---|---|---|---|
> | DWP（期刊版，γ = 0）| github.com/BailiangJ/rethink-reg @97175ff（PyTorch）| 原作程式不改，`ASD\reproduce\train_dwp.py` 只換資料 | 讀我們的 npz、受試者 → atlas、影像 192×224×192、評估用我們的 |
> | RCN | github.com/microsoft/Recursive-Cascaded-Networks @0709490（TensorFlow 1.4）| `ASD\reproduce\rcn_torch.py` 逐行翻成 PyTorch（筆電驗證跟原作逐項相同）| 框架、影像大小、讀我們的 npz、atlas 換 MNI152、評估 |
> | TransMorph | 官方 IXI 的 `train_TransMorph.py` | TransMorph 專案 `mixv2\train_mixv2.py`（預設＝官方訓練方式）| 影像大小與 Swin 視窗、受試者 → atlas（官方 atlas → 受試者）、評估 |
> | TransMorph-diff | 官方 IXI 的 `train_TransMorph_diff.py` | TransMorph 專案 `mixv2\train_mixv2_diff.py`（預設＝官方）| 同上 |
>
> **訓練量不必照原作**（使用者 10-28）。下面每一顆都給兩種：照原作的量，或縮短成跟其他模型一樣約 2.5 萬步；學習率排程照原作的形狀跟著總步數縮。
> 跑的機器：**AI**。🔴 **一次只跑一顆**。指令全部是 **cmd（命令提示字元）** 語法。最後更新：2026-10-28（筆電：四支都試跑過、評估跑通）

---

## cmd 要注意的三件事

1. **每個指令都是一整行**：整行複製、貼上、按 Enter。看起來換行的地方是畫面自動折行，不是真的斷開
2. **換磁碟要用 `cd /d`**：只打 `cd D:\...` 的話，如果目前在 C 槽，會停在原地不動
3. 虛擬環境啟動成功的話，提示字元前面會多一個 `(venv_dwp)`、`(vxm_env)` 或 `(venv)`

---

## 0. 第一次（只要做一次）

### 0-1 抓原作程式（固定在讀過的那個版本）

```bat
mkdir D:\chengyun\orig_repos
```

```bat
cd /d D:\chengyun\orig_repos
```

```bat
git clone https://github.com/BailiangJ/rethink-reg.git
```

```bat
git -C rethink-reg checkout 97175ff
```

RCN 的原作只拿來對照（訓練用的是翻好的 PyTorch 版），要看再抓：

```bat
git clone https://github.com/microsoft/Recursive-Cascaded-Networks.git
```

### 0-2 DWP 的環境 venv_dwp（套件版本照原作的 requirements.txt）

先查 vxm_env 用的是哪一個 Python（看 `home = ...` 那一行，下面的 `<home 路徑>` 換成它）：

```bat
type D:\chengyun\TRY_voxelmorph\vxm_env\pyvenv.cfg
```

```bat
virtualenv D:\chengyun\orig_repos\venv_dwp -p "<home 路徑>\python.exe"
```

（`virtualenv` 找不到的話，同 TransMorph 操作單：改打 `"<home 路徑>\python.exe" -m virtualenv D:\chengyun\orig_repos\venv_dwp`）

```bat
D:\chengyun\orig_repos\venv_dwp\Scripts\activate.bat
```

```bat
python -m pip install --upgrade pip
```

torch 照原作用 2.5.1（CUDA 12.1 版；AI 的驅動最高 CUDA 12.2，可以用）：

```bat
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

```bat
python -m pip install -r D:\chengyun\TRY_voxelmorph\ASD\reproduce\requirements_dwp.txt
```

確認（最後要印出 `2.5.1+cu121 True 1.5.0`）：

```bat
python -c "import torch, monai; print(torch.__version__, torch.cuda.is_available(), monai.__version__)"
```

```bat
deactivate
```

---

## 每一顆開始前

```bat
cd /d D:\chengyun\TRY_voxelmorph
```

```bat
git pull
```

確認顯卡空出來了（沒有其他 python 在用 GPU）：

```bat
nvidia-smi
```

---

## A. DWP（`models\orig_dwp`）

| | 照原作 | 縮短 |
|---|---|---|
| 訓練量 | 100 輪 × 418 步 = 41,800 步（lr 每輪 × 0.996）| 60 輪 = 25,080 步（加 `--max-epochs 60`）|
| 時間（筆電實測每步 12 秒；AI 應該差不多或更快）| 約 6 天 | 約 3.5 天 |

顯存約 6 GB（筆電實測）。每輪存一顆 `NNNN.pth`（約 7.5 MB）。

⚠️ 原作設了 `cudnn.deterministic = True`，筆電上反向傳播因此慢 5.6 倍。只影響速度和「同一個 seed 逐位元相同」，方法不變；
嫌太慢可以在訓練指令最後加 `--nondeterministic`（筆電每步約 3.6 秒，照原作的量約 42 小時）。

**A-1 訓練**：

```bat
D:\chengyun\orig_repos\venv_dwp\Scripts\activate.bat
```

照原作：

```bat
python ASD\reproduce\train_dwp.py data\mixed_preprocessed_v2\train --atlas IXI\atlas_mni152_09c_v3.npz --model-dir models\orig_dwp --orig-repo D:\chengyun\orig_repos\rethink-reg --gpu 0
```

或縮短：

```bat
python ASD\reproduce\train_dwp.py data\mixed_preprocessed_v2\train --atlas IXI\atlas_mni152_09c_v3.npz --model-dir models\orig_dwp --orig-repo D:\chengyun\orig_repos\rethink-reg --max-epochs 60 --gpu 0
```

- log 在 `models\orig_dwp\train_log.txt`；第二行要看到 `VXMPlus_DualWarpPy，1886796 個參數；影像 192x224x192；418 位受試者 → atlas`
- 一開始的 loss 約 −0.3（含粗的層），括號裡第一個數（最細那層的局部 NCC，公式同 VoxelMorph）約 −0.07～−0.08，跟 mix_exp6 第一步一樣

**A-2 用 val 挑 epoch**（🔴 不要拿 test 挑）：

```bat
python ASD\reproduce\eval_repro.py --method dwp --model-dir models\orig_dwp --step 5 --test-dir data\mixed_preprocessed_v2\val --orig-repo D:\chengyun\orig_repos\rethink-reg --gpu 0
```

取 `models\orig_dwp\dice_curve_val.csv` 裡 `dice_mean` 最大的 epoch。

**A-3 test（只跑一次）**，`NNNN` 換成挑出來的 epoch：

```bat
python ASD\reproduce\eval_repro.py --method dwp --model models\orig_dwp\NNNN.pth --test-dir data\mixed_preprocessed_v2\test --orig-repo D:\chengyun\orig_repos\rethink-reg --surface --gpu 0
```

```bat
deactivate
```

---

## B. RCN（`models\orig_rcn_vtn3`）

底層網路與顆數原作可以選（`-b`、`-n`），論文放出來的腦部模型是 VTN 3 顆、VTN 10 顆、VoxelMorph 5 顆。
下面用 **VTN 3 顆**（筆電實測約 7 GB；batch 4 的 4 對是依序算、梯度平均，顯存跟 1 對一樣。
VoxelMorph 底在我們的影像大小每顆約 4～5 GB，5 顆放不進 24 GB）。

| | 照原作 | 縮短 |
|---|---|---|
| 訓練量 | batch 4 × 10 萬步，lr 6 萬、8 萬步各減半 | 2.5 萬步，lr 1.5 萬、2 萬步各減半（加 `--round 5000`）|
| 時間（一步 4 對；筆電實測 VTN 1 顆每對 0.27 秒、2 顆 0.5 秒、3 顆 1.2～1.5 秒（筆電顯存快滿，AI 估計約 0.75 秒））| 約 3.5～7 天 | 約 1～2 天 |

每 5,000 步存一顆 `NNNNNN.pt`（約 380 MB；照原作 20 顆、縮短 5 顆 → 縮短的話加 `--save-every 1000`，25 顆）；
`last.pt` 是最新的一顆加上 Adam 狀態，中斷了可以用 `-c models\orig_rcn_vtn3\last.pt` 接著練。

**B-1 訓練**：

```bat
vxm_env\Scripts\activate.bat
```

照原作：

```bat
python ASD\reproduce\train_rcn.py -b VTN -n 3 --train-dir data\mixed_preprocessed_v2\train --atlas IXI\atlas_mni152_09c_v3.npz --model-dir models\orig_rcn_vtn3 -g 0
```

或縮短：

```bat
python ASD\reproduce\train_rcn.py -b VTN -n 3 --train-dir data\mixed_preprocessed_v2\train --atlas IXI\atlas_mni152_09c_v3.npz --model-dir models\orig_rcn_vtn3 --round 5000 --save-every 1000 -g 0
```

- 第二行要看到 `VTN × 3 顆（＋affine），rep 1，共 99094929 個參數；影像 192x224x192；batch 4`
- 一開始平滑項（括號第二個數）很大（約 50），是原作 VTN 的初始化（輸出 ×20），幾百步內會掉下來，正常

**B-2 用 val 挑**（縮短的話 `--step 1`，照原作 `--step 1` 也只有 20 顆）：

```bat
python ASD\reproduce\eval_repro.py --method rcn --model-dir models\orig_rcn_vtn3 --step 1 --test-dir data\mixed_preprocessed_v2\val --gpu 0
```

**B-3 test（只跑一次）**，`NNNNNN` 換成挑出來的步數：

```bat
python ASD\reproduce\eval_repro.py --method rcn --model models\orig_rcn_vtn3\NNNNNN.pt --test-dir data\mixed_preprocessed_v2\test --surface --gpu 0
```

---

## C. TransMorph（TransMorph 專案，`runs\tm_orig`）

| | 照原作 | 縮短 |
|---|---|---|
| 訓練量 | 500 輪 × 418 步 ≈ 21 萬步（lr poly 遞減）| 60 輪 = 25,080 步（加 `--epochs 60`）|
| 時間（tm_mixv2 每步約 2.5 秒）| 約 6 天 | 約 17～21 小時 |

每 5 輪存一顆（約 187 MB；照原作 100 顆約 19 GB、縮短 12 顆約 2.3 GB）。

```bat
cd /d D:\chengyun\Transmorph_NewArchitecture
```

```bat
git pull
```

```bat
venv\Scripts\activate.bat
```

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85
```

照原作（同一個視窗）：

```bat
python mixv2\train_mixv2.py D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\train --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --model-dir runs\tm_orig --gpu 0
```

或縮短：

```bat
python mixv2\train_mixv2.py D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\train --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --model-dir runs\tm_orig --epochs 60 --gpu 0
```

- 第二行要看到 `每輪 全部洗牌跑一次（418 步）` 和 `Adam amsgrad … poly 0.9 遞減；翻轉 開`

val 挑 epoch、test（同 tm_diff_mixv2 的操作單，只換資料夾名）：

```bat
python mixv2\eval_mixv2.py --model-dir runs\tm_orig --step 5 --test-dir D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\val --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --atlas-seg D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3_seg.npz --gpu 0
```

```bat
python mixv2\eval_mixv2.py --model runs\tm_orig\NNNN.pt --test-dir D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\test --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --atlas-seg D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3_seg.npz --surface --gpu 0
```

---

## D. TransMorph-diff（TransMorph 專案，`runs\tm_diff_orig`）

訓練量、時間、存檔同 C。環境同 C（`venv\Scripts\activate.bat`、`set PYTORCH_CUDA_ALLOC_CONF=...`）。

照原作：

```bat
python mixv2\train_mixv2_diff.py D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\train --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --model-dir runs\tm_diff_orig --gpu 0
```

或縮短：

```bat
python mixv2\train_mixv2_diff.py D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\train --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --model-dir runs\tm_diff_orig --epochs 60 --gpu 0
```

val、test 同 C，把 `runs\tm_orig` 換成 `runs\tm_diff_orig`。

---

## 跑完之後

把每一顆的 `train_log.txt`、`dice_curve_val.csv`、`dice_NNNN.csv`、`surface_NNNN.csv` 帶回筆電（.pt／.pth 不用）：
`models\orig_dwp\`、`models\orig_rcn_vtn3\`；TransMorph 兩顆放筆電的 `models\tm_orig\`、`models\tm_diff_orig\`。
