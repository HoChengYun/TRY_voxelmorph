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
> **訓練量不必照原作**（使用者 10-28）。下面每一顆都給兩種：照原作的量，或縮短成跟其他模型一樣約 2.5 萬步；學習率排程照原作的形狀跟著總步數縮（DWP 例外，見 A 節）。
> 跑的機器：**AI**。🔴 **一次只跑一顆**。指令全部是 **cmd（命令提示字元）** 語法。最後更新：2026-10-30（DWP 訓練量定案：60 輪＋`--nondeterministic`；RCN 定案 VoxelMorph × 2 顆、6,250 步（`--round 1250`））；10-28 筆電四支都試跑過、評估跑通

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

**訓練量（10-30 使用者決定）：60 輪＋`--nondeterministic`**（使用者：「反正我們資料集和他不一樣」）

| | 用這個 |
|---|---|
| 訓練量 | 60 輪 × 418 步 = 25,080 步 ≈ 原作的 25,000 步（`--max-epochs 60`）|
| 時間（筆電每步 3.6 秒；AI 應該差不多或更快）| 約 25 小時 |

- 為什麼是 60 輪：原作每一步用 2 顆互配，LUMIR 500 顆 → 每輪 250 步，100 輪 = 25,000 步。我們每一步是 1 位受試者配 atlas、每輪 418 步，跑 100 輪會是原作的 1.67 倍。lr 每輪 × 0.996 照原作，60 輪後降到 0.79 倍（原作 100 輪後 0.67 倍）（手冊 §31.4）
- `--nondeterministic`：不設原作的 `cudnn.deterministic = True`。原作設它是為了同一個 seed 每次逐位元相同，代價是筆電上每步 12 秒（不設 3.6 秒）；方法不變
- 顯存約 6 GB（筆電實測）。每輪存一顆 `NNNN.pth`（約 7.5 MB），60 顆約 450 MB

**A-1 訓練**：

```bat
D:\chengyun\orig_repos\venv_dwp\Scripts\activate.bat
```

```bat
python ASD\reproduce\train_dwp.py data\mixed_preprocessed_v2\train --atlas IXI\atlas_mni152_09c_v3.npz --model-dir models\orig_dwp --orig-repo D:\chengyun\orig_repos\rethink-reg --max-epochs 60 --nondeterministic --gpu 0
```

- log 在 `models\orig_dwp\train_log.txt`；第二行要看到 `VXMPlus_DualWarpPy，1886796 個參數；影像 192x224x192；418 位受試者 → atlas`，同一行後面是 `每輪 418 步、60 輪；seed 2023；cudnn.deterministic False`
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

## B. RCN（`models\orig_rcn_vxm2`）

**底層網路與顆數、訓練量（10-30 使用者決定）：VoxelMorph × 2 顆，6,250 步（`--round 1250`）**

- 原作可以選底層網路與顆數（`-b`、`-n`），論文放出來的腦部模型是 VTN 3 顆、VTN 10 顆、VoxelMorph 5 顆。VoxelMorph 每顆約 4～5 GB，5 顆在我們的影像大小放不進 24 GB
- 用 2 顆：跟 mix_cascade、mix_cascade_rcn 一樣串兩顆，可以直接比「原作程式 vs 我們自己寫的」。RCN 的 VoxelMorph 是它自己寫的 VoxelMorph-2（編碼 16-32-32-32、解碼 32-32-32-32-32-16-16，位移場、不積分），每顆大小跟 mix_exp6 一樣；前面另有原作的 affine 網路

| | 用這個 |
|---|---|
| 訓練量 | batch 4 × 6,250 步（`--round 1250`）= 25,000 對，看過的影像對數同其他模型（25,000 步 × 1 對）；lr 1e-4，3,750、5,000 步各減半。原作 10 萬步、6 萬／8 萬步減半，形狀一樣、整個縮成 1/16 |
| 顯存 | 筆電實測 11.3 GB（batch 4 的 4 對是依序算、梯度平均，顯存跟 1 對一樣）|
| 時間 | 10-30 跟 DWP 同時跑，實測每步 8.17 秒（每對約 2 秒）→ 約 14 小時 |

`--round 1250`：原作把訓練分成 5 段，每段 `--round` 步（預設 20,000），lr 在第 3、4 段開頭各減半；改成 1,250 就是總共 6,250 步。代價：權重只更新 6,250 次，是原作的 1/16，分數偏低時可能是練不夠、不一定是方法差。
`--save-every 250`：每 250 步存一顆 `NNNNNN.pt`（預設 5,000 步，縮短後只會有 2 顆可挑），約 60 MB，共 25 顆；
`last.pt` 是最新的一顆加上 Adam 狀態，中斷了可以用 `-c models\orig_rcn_vxm2\last.pt` 接著練。

**B-1 訓練**（開一個新的 cmd）：

```bat
cd /d D:\chengyun\TRY_voxelmorph
```

```bat
vxm_env\Scripts\activate.bat
```

跟 DWP 同時跑的話，先設顯存上限（RCN 最多用約 14 GB）。🔴 只設在 RCN 這個視窗：DWP 的 torch 2.5.1 看不懂這個設定，會直接報錯
（`Unrecognized CachingAllocator option: per_process_memory_fraction`）。單獨跑就不用設：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.6
```

```bat
python ASD\reproduce\train_rcn.py -b VoxelMorph -n 2 --train-dir data\mixed_preprocessed_v2\train --atlas IXI\atlas_mni152_09c_v3.npz --model-dir models\orig_rcn_vxm2 --round 1250 --save-every 250 -g 0
```

- 第二行要看到 `VoxelMorph × 2 顆（＋affine），rep 1，共 14939414 個參數；影像 192x224x192；batch 4；6250 步（lr 0.0001，3750、5000 步後各減半）`
- 之後每行的 `step: N/250` 是「這一段 250 步裡的第 N 步」（每 250 步存一次檔，共 25 段），`epoch` 是第幾段
- 跳出 `CUDA out of memory` 的話，上限改成 `0.65` 再跑一次
- 兩顆都開跑後看一下 `nvidia-smi`：顯存合計要在 24 GB 以內；DWP 的 log 每步如果變成 10 幾秒，就是顯存溢位，先停掉 RCN

**B-2 用 val 挑**（25 顆都掃，`--step 1`）：

```bat
python ASD\reproduce\eval_repro.py --method rcn --model-dir models\orig_rcn_vxm2 --step 1 --test-dir data\mixed_preprocessed_v2\val --gpu 0
```

**B-3 test（只跑一次）**，`NNNNNN` 換成挑出來的步數：

```bat
python ASD\reproduce\eval_repro.py --method rcn --model models\orig_rcn_vxm2\NNNNNN.pt --test-dir data\mixed_preprocessed_v2\test --surface --gpu 0
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
`models\orig_dwp\`、`models\orig_rcn_vxm2\`；TransMorph 兩顆放筆電的 `models\tm_orig\`、`models\tm_diff_orig\`。
