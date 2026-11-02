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
> 跑的機器：**AI**。🔴 **一次只跑一顆**。指令全部是 **cmd（命令提示字元）** 語法。最後更新：2026-11-02（加 B3：RCN 程式跑 1 顆 VoxelMorph）；2026-10-31（加 B2：RCN 往同一位受試者 sub-0036）；2026-10-30（DWP 訓練量定案：60 輪＋`--nondeterministic`；RCN 定案 VTN × 3 顆、2.5 萬步（`--round 5000`））；10-28 筆電四支都試跑過、評估跑通

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

> 🔴 11-02 更正：`--nondeterministic` **實際沒有生效**。`train_dwp.py` 先設 False，接著原作的 `set_seed()`（`utils/__init__.py:33`）又改回 True，所以是照原作的 deterministic 跑的。只影響速度與逐位元重現，方法與分數不受影響。

| | 用這個 |
|---|---|
| 訓練量 | 60 輪 × 418 步 = 25,080 步 ≈ 原作的 25,000 步（`--max-epochs 60`）|
| 時間 | AI 實測約 29 小時（大多跟別的訓練同時跑，每步中位數 4.38 秒）|

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

- log 在 `models\orig_dwp\train_log.txt`；第二行要看到 `VXMPlus_DualWarpPy，1886796 個參數；影像 192x224x192；418 位受試者 → atlas`，同一行後面是 `每輪 418 步、60 輪；seed 2023；cudnn.deterministic True`
- 一開始的 loss 約 −0.3（含粗的層），括號裡第一個數（最細那層的局部 NCC，公式同 VoxelMorph）約 −0.07～−0.08，跟 mix_exp6 第一步一樣

**A-2 用 val 挑 epoch**（🔴 不要拿 test 挑）：

```bat
python ASD\reproduce\eval_repro.py --method dwp --model-dir models\orig_dwp --step 1 --test-dir data\mixed_preprocessed_v2\val --orig-repo D:\chengyun\orig_repos\rethink-reg --gpu 0
```

取 `models\orig_dwp\dice_curve_val.csv` 裡 `dice_mean` 最大的 epoch。第一次跑是 `--step 5`，只掃 13 顆、挑到第 51 輪；11-02 改成 `--step 1` 全部掃。

**A-3 test（只跑一次）**，`NNNN` 換成挑出來的 epoch：

```bat
python ASD\reproduce\eval_repro.py --method dwp --model models\orig_dwp\NNNN.pth --test-dir data\mixed_preprocessed_v2\test --orig-repo D:\chengyun\orig_repos\rethink-reg --surface --gpu 0
```

```bat
deactivate
```

---

## B. RCN（`models\orig_rcn_vtn3`）

**底層網路與顆數、訓練量（10-30 使用者決定）：VTN × 3 顆，2.5 萬步（`--round 5000`）**

- 原作可以選底層網路與顆數（`-b`、`-n`），預設 VTN；論文放出來的腦部模型是 VTN 3 顆、VTN 10 顆、VoxelMorph 5 顆 → 用 **VTN 3 顆**，跟原作放出的模型同一種設定
- VTN（Volume Tweening Network，RCN 同一群作者）不是 VoxelMorph 的變形：參考 FlowNet，往下縮 6 層（到 1/64）、每格最多 512 個數字，每一層都先猜粗的形變再往上修細；一顆約 2,800 萬參數，但大的層都在很小的格子上，顯存反而比 VoxelMorph 少。最前面的 affine 網路也是 VTN 論文的
- 代價：底層網路跟 mix_cascade（我們的 VoxelMorph 串兩顆）不同，不能直接拿來比「原作程式 vs 我們的實作」
- 10-30 先開跑的 VoxelMorph × 2（`models\orig_rcn_vxm2`）已停掉，不用

| | 用這個 |
|---|---|
| 訓練量 | batch 4 × 2.5 萬步（`--round 5000`）：權重更新次數同其他模型（mix_exp6、DWP 都是 2.5 萬步），看過的影像對數 10 萬對（其他模型的 4 倍）；lr 1e-4，1.5 萬、2 萬步各減半。原作 10 萬步、6 萬／8 萬步減半，形狀一樣、整個縮成 1/4 |
| 顯存 | 筆電實測約 7 GB（batch 4 的 4 對是依序算、梯度平均，顯存跟 1 對一樣）|
| 時間 | 10-30 跟 DWP 同時跑實測每步 2.59 秒（每對約 0.65 秒）→ 約 18 小時 |

`--round 5000`：原作把訓練分成 5 段（`--epochs`，預設 5），每段 `--round` 步（預設 20,000），lr 在第 4、5 段開頭各減半（前 3 段 1e-4、第 4 段 5e-5、第 5 段 2.5e-5）；改成 5,000 就是總共 2.5 萬步（原作的 1/4）。10-30 先用 `--round 1250`（6,250 步）開跑，跑起來比預期快，同日改成這個。
`--save-every 1000`：每 1,000 步存一顆 `NNNNNN.pt`（預設 5,000 步，縮短後只會有 5 顆可挑），**約 380 MB，共 25 顆約 9.5 GB**；
`last.pt` 是最新的一顆加上 Adam 狀態（約 1.2 GB），中斷了可以用 `-c models\orig_rcn_vtn3\last.pt` 接著練（其他參數要跟原本一樣）。

**B-1 訓練**（開一個新的 cmd）：

```bat
cd /d D:\chengyun\TRY_voxelmorph
```

```bat
vxm_env\Scripts\activate.bat
```

跟 DWP 同時跑的話，先設顯存上限（RCN 最多用約 12 GB）。🔴 只設在 RCN 這個視窗：DWP 的 torch 2.5.1 看不懂這個設定，會直接報錯
（`Unrecognized CachingAllocator option: per_process_memory_fraction`）。單獨跑就不用設：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.5
```

```bat
python ASD\reproduce\train_rcn.py -b VTN -n 3 --train-dir data\mixed_preprocessed_v2\train --atlas IXI\atlas_mni152_09c_v3.npz --model-dir models\orig_rcn_vtn3 --round 5000 --save-every 1000 -g 0
```

- 第二行要看到 `VTN × 3 顆（＋affine），rep 1，共 99094929 個參數；影像 192x224x192；batch 4；25000 步（lr 0.0001，15000、20000 步後各減半）`
- 之後每行的 `step: N/1000` 是「這一段 1,000 步裡的第 N 步」（每 1,000 步存一次檔，共 25 段），`epoch` 是第幾段
- 一開始平滑項（括號第二個數）很大（約 50），是原作 VTN 的初始化（輸出 ×20），幾百步內會掉下來，正常
- 跳出 `CUDA out of memory` 的話，上限改成 `0.6` 再跑一次
- 兩顆都開跑後看一下工作管理員：「共用 GPU 記憶體」要維持在 1 GB 以下；DWP 的 log 每步如果變成 10 幾秒，就是顯存溢位，先停掉 RCN

**B-2 用 val 挑**（25 顆都掃，`--step 1`）：

```bat
python ASD\reproduce\eval_repro.py --method rcn --model-dir models\orig_rcn_vtn3 --step 1 --test-dir data\mixed_preprocessed_v2\val --gpu 0
```

**B-3 test（只跑一次）**，`NNNNNN` 換成挑出來的步數：

```bat
python ASD\reproduce\eval_repro.py --method rcn --model models\orig_rcn_vtn3\NNNNNN.pt --test-dir data\mixed_preprocessed_v2\test --surface --gpu 0
```

---

## B2. RCN 往同一位受試者（`models\orig_rcn_vtn3_subj`）

**10-31 使用者決定**：B 節那顆的小腦白質低於 affine（0.717 → 0.640），懷疑是 MNI152 的小腦太亮（小腦白質、皮質、腦幹都比 51 位 test 受試者亮）。
原作的 atlas 是 LPBA 的一位真人，這顆照原作改成一位受試者當 atlas，其他都跟 B 節一樣。**不用改程式**。

| | 用這個 |
|---|---|
| atlas | `sub-0036`（val 裡亮度、腦大小最接近中位數，影像邊緣乾淨；npz 裡有 vol、seg，`--atlas`、`--atlas-seg` 都指向它）|
| 訓練 | 跟 B 節相同：train 418 位、VTN × 3、batch 4、`--round 5000 --save-every 1000` |
| val | 50 位（扣掉 sub-0036 自己，不然會算到自己對自己）|
| test | 51 位，不變 |
| 時間 | B 節實際約 16 小時（跟 DWP 同時跑）|
| 硬碟 | 同 B 節：25 顆約 9.5 GB，`last.pt` 約 1.2 GB |

🔴 **Dice 不能跟 atlas 是 MNI152 的模型直接比**：標準答案換成 sub-0036 的標籤。要跟這個 atlas 的 affine 起點比：30 個結構 0.6499、小腦白質 0.736、小腦皮質 0.784（筆電已算好，`models\orig_rcn_vtn3_subj\dice_baseline.csv`）。

**B2-1 訓練**（開一個新的 cmd）：

```bat
cd /d D:\chengyun\TRY_voxelmorph
```

```bat
vxm_env\Scripts\activate.bat
```

DWP 還在跑的話，先設顯存上限（只設在這個視窗）；單獨跑就不用設：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.5
```

```bat
python ASD\reproduce\train_rcn.py -b VTN -n 3 --train-dir data\mixed_preprocessed_v2\train --atlas data\mixed_preprocessed_v2\val\sub-0036.npz --model-dir models\orig_rcn_vtn3_subj --round 5000 --save-every 1000 -g 0
```

- 第二行要看到 `VTN × 3 顆（＋affine），rep 1，共 99094929 個參數；影像 192x224x192；batch 4；25000 步`，後面是 `418 位受試者 → atlas`
- 第一行的指令要看得到 `val\sub-0036.npz`

**B2-2 用 val 挑**：先做一份不含 sub-0036 的 val（複製 50 個檔；robocopy 最後 `Copied` 那一列的 Files 要是 50）：

```bat
robocopy data\mixed_preprocessed_v2\val data\rcn_sub0036\val *.npz /XF sub-0036.npz
```

```bat
python ASD\reproduce\eval_repro.py --method rcn --model-dir models\orig_rcn_vtn3_subj --step 1 --test-dir data\rcn_sub0036\val --atlas data\mixed_preprocessed_v2\val\sub-0036.npz --atlas-seg data\mixed_preprocessed_v2\val\sub-0036.npz --gpu 0
```

**B2-3 test（只跑一次）**，`NNNNNN` 換成挑出來的步數：

```bat
python ASD\reproduce\eval_repro.py --method rcn --model models\orig_rcn_vtn3_subj\NNNNNN.pt --test-dir data\mixed_preprocessed_v2\test --atlas data\mixed_preprocessed_v2\val\sub-0036.npz --atlas-seg data\mixed_preprocessed_v2\val\sub-0036.npz --surface --gpu 0
```

**怎麼看結果**（要帶回 `NNNNNN.pt`，才算得出小腦體積）：

- 小腦白質高於 affine 的 0.736，體積接近 1 倍 → MNI152 的亮度是主因
- 小腦白質還是低於 affine、體積還是被撐大 → 問題不只在 atlas

---

## B3. RCN 程式跑 1 顆 VoxelMorph（`models\orig_rcn_vxm1`）

**11-02 使用者決定**：要看 B 節那顆的小腦問題跟 VTN 有沒有關係。這顆跟 B 節只差底層網路（3 顆 VTN → 1 顆 VoxelMorph），就是 RCN 論文表 2 的 `1-cascade VoxelMorph`。**不用改程式**。

| | 用這個 |
|---|---|
| atlas | MNI152（同 B 節）|
| 相似度 | 全域相關係數，整張影像算一個數字（同 B 節；跟 mix_exp3、DWP 的局部 NCC 不同）|
| 網路 | 最前面的 affine（1,434 萬參數）＋ RCN 版 VoxelMorph（30 萬參數），共 14,638,867 個參數 |
| 訓練量 | 同 B 節：batch 4、2.5 萬步（`--round 5000`），lr 在 1.5 萬、2 萬步各減半 |
| 存檔 | 每 1,000 步一顆 `NNNNNN.pt`（估計約 60 MB），共 25 顆約 1.5 GB；`last.pt` 另含 Adam 狀態 |
| 顯存 | 筆電試跑實際用到 6.76 GB、PyTorch 預留 9.83 GB（筆電 8 GB 放不下，每步 1～3 分鐘，要在 AI 跑） |
| 時間 | 估計約 28 小時（VoxelMorph 2 顆跟 DWP 同時跑每步 8.17 秒，1 顆約一半；沒實測）|

🔴 它雖然叫 VoxelMorph，但不是 mix_exp3：相似度、affine、資料增強、batch、lr 都跟 RCN 一樣。

論文最多串 5 顆（LPBA 0.708），但我們的影像較大，5 顆估計約 25 GB，AI 放不下。想跟 B 節只差底層網路，可以改成 3 顆：指令改 `-n 3`、`--model-dir models\orig_rcn_vxm3`（顯存估計約 16 GB、約 3.5 天，要單獨跑）。

**B3-1 訓練**（開一個新的 cmd）：

```bat
cd /d D:\chengyun\TRY_voxelmorph
```

```bat
vxm_env\Scripts\activate.bat
```

B2 或其他程式還在用 GPU 的話，先設顯存上限（只設在這個視窗，0.4 約 9.6 GB，夠這顆用）。單獨跑就不用設：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.4
```

```bat
python ASD\reproduce\train_rcn.py -b VoxelMorph -n 1 --train-dir data\mixed_preprocessed_v2\train --atlas IXI\atlas_mni152_09c_v3.npz --model-dir models\orig_rcn_vxm1 --round 5000 --save-every 1000 -g 0
```

- 第二行要看到 `VoxelMorph × 1 顆（＋affine），rep 1，共 14638867 個參數；影像 192x224x192；batch 4；25000 步（lr 0.0001，15000、20000 步後各減半）`
- 跳出 `CUDA out of memory` 的話，把上限調高 0.05 再跑一次
- 每步如果變成 10 幾秒以上，看工作管理員的「共用 GPU 記憶體」，超過 1 GB 就是顯存溢位，先停掉一顆

**B3-2 用 val 挑**（25 顆都掃，`--step 1`）：

```bat
python ASD\reproduce\eval_repro.py --method rcn --model-dir models\orig_rcn_vxm1 --step 1 --test-dir data\mixed_preprocessed_v2\val --gpu 0
```

**B3-3 test（只跑一次）**，`NNNNNN` 換成挑出來的步數：

```bat
python ASD\reproduce\eval_repro.py --method rcn --model models\orig_rcn_vxm1\NNNNNN.pt --test-dir data\mixed_preprocessed_v2\test --surface --gpu 0
```

**怎麼看結果**（要帶回 `NNNNNN.pt`，才算得出每個結構的體積）。比較對象：affine 小腦白質 0.717；B 節 RCN 0.640、體積 1.72 倍；DWP 0.827、體積 1.00 倍：

- 小腦白質接近 0.64、體積也被撐大 → 不是 VTN 的問題，指向全域相關係數配 MNI152 atlas
- 小腦白質接近 0.82、體積接近 1 倍 → 跟 VTN 或串 3 顆有關（這顆同時換了網路和顆數，分不出是哪一個）
- 落在中間 → 兩邊都有影響，要再拆開

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
`models\orig_dwp\`、`models\orig_rcn_vtn3\`、`models\orig_rcn_vtn3_subj\`、`models\orig_rcn_vxm1\`（這兩顆另外帶回挑出來的 `NNNNNN.pt`，要分析小腦）；TransMorph 兩顆放筆電的 `models\tm_orig\`、`models\tm_diff_orig\`。
