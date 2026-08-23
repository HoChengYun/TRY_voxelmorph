# 前一個 AI 擔心的事

> 寫給下一個接手的 AI（以及未來的我自己）。
> 產出日期：**2026-08-23**
> 作者：一個被叫來「檢查筆記有沒有漏講」的 Claude
> ⚠️ **我全程只讀沒改。** 底下所有「建議改法」都還沒執行，請先跟使用者確認再動手。

---

## 0. 這份文件在幹嘛

使用者請我檢查 `CLAUDE.md`、`VoxelMorph_PyTorch_實作指南.md`、`IXI/ixi相關手冊.md`
有沒有漏掉的東西。我掃了整個工作區跟三份文件對帳，結果是**漏的比想像中多，
而且有幾處不只是過時，是會讓人做錯事**。

我沒有修改任何檔案（使用者還沒點頭）。這份清單就是留給你去修的。

**修之前請先做兩件事：**
1. 問使用者要修哪些（有些牽涉到他才知道的實驗參數）
2. 每一項我都附了「怎麼自己驗證」的指令，別直接相信我，重跑一次

---

## 1. 最重要的一件事：文件的新舊順序跟直覺相反

使用者原本以為「後期的筆記都在 `IXI/ixi相關手冊.md`」。**實際上剛好相反。**

| 文件 | 內容停在哪個時間點 | 新舊 |
|------|-------------------|------|
| `VoxelMorph_PyTorch_實作指南.md` | 2026/03，OASIS 時期 | 最舊 |
| `IXI/ixi相關手冊.md` | 2026/04，v2 / resample 時期 | 中間 |
| `CLAUDE.md` | 2026/08，v3 / exp8 | **最新** |

`CLAUDE.md` 自己也寫著「先讀 `ixi相關手冊.md`（操作細節最完整）」——
這句話現在會把人帶去讀一份停在 v2 的文件。

> 🔴 **給下一個 AI 的實質建議**：把 `CLAUDE.md` 定為唯一事實來源（single source of truth），
> 另外兩份在開頭各加一段警語，標明它記錄的是哪個時間點的狀態。
> 但**光加警語不夠**，第 2 節那幾處必須實際改掉。

---

## 2. 🔴 會主動誤導的部分（優先修）

### 2.1 手冊 §6.1 直接否定了現行做法

`IXI/ixi相關手冊.md` 第 410 行附近的表格寫著：

> | 裁切（cropping） | 砍掉邊緣 voxel | ❌ 可能砍到腦組織邊緣 |

但 **v3（現行版本）用的就是 `--method crop`**。`CLAUDE.md` 的資料版本表寫得很清楚：
v3 是 `--method crop`、spacing 精確 1mm、現行使用中。

照手冊做會退回 v2。這是整份文件最危險的一處。

**建議改法**：把那一列改成 ✅ 並註明「v3 起改用此法，裁掉的是背景 voxel，spacing 維持精確 1mm」，
`resample` 那列改註「v2 使用，spacing 會變成 ≈(1.005, 1.022, 1.005)mm」。

---

### 2.2 手冊完全沒有 v1 / v2 / v3 的概念

手冊裡的 atlas 檔名是 `atlas_mni152_09c.nii.gz` 和 `atlas_mni152_09c_resize.npz`，
**這兩個檔案都不存在**。

實際存在的（我 `ls` 過）：

```
IXI/atlas_mni152_09c_v2.nii.gz
IXI/atlas_mni152_09c_v2.npz
IXI/atlas_mni152_09c_v3.nii.gz
IXI/atlas_mni152_09c_v3.npz
```

注意 **v1 的 atlas 檔案已經不在了**。這代表 `IXI_preprocessed`（v1）那批資料
實際上已經**不可重現**——想重跑得先重建 atlas，但當初用的是早期 `scipy.ndimage.zoom`。
`CLAUDE.md` 說 v1「不要再用」是對的，但沒說「而且回不去了」。

更關鍵的是，`CLAUDE.md` 那條鐵則——

> 🔴 訓練用的 `--atlas` 版本必須跟 `datadir` 的資料版本一致

——**手冊完全沒有**。這是最容易炸的坑，舊文件卻沒警告。

**驗證指令**：
```bash
ls IXI/atlas*.nii.gz IXI/atlas*.npz
ls -d IXI/IXI_preprocessed*
```

---

### 2.3 `--target-shape` 已經移除，手冊還在教人用

`CLAUDE.md` 明寫：

> ⚠️ `preprocess_ixi.py` 的 `--target-shape` **已移除**，輸出 shape 直接由 `--atlas` 決定。

但手冊有兩處還在用它：
- §4.5 參數表列出 `--target-shape | 192,224,192 | 輸出影像大小`
- §6 常見問題：「`RuntimeError: size XXX not divisible` → 在 preprocess_ixi.py 指定 `--target-shape` 修正」

第二處尤其糟——它給的是一個**現在會直接報 unrecognized argument 的解法**。

**修之前先確認**（我沒實際跑腳本）：
```bash
python IXI/preprocess_ixi.py --help
```

---

### 2.4 手冊 §5.1 的訓練指令跑不動

手冊寫的是：

```powershell
python voxelmorph-code\scripts\torch\train.py \
    --datadir  IXI\IXI_preprocessed\train \
    ...
```

兩個問題：
1. `--datadir` 不是旗標，datadir 是**位置參數**（見 `CLAUDE.md` 的常用指令）
2. PowerShell 區塊卻用 `\` 換行，要用反引號 `` ` ``

---

### 2.5 手冊 §9.3 說「若要修改，在 train.py 加 argument」——早就做完了

手冊建議自己去 train.py 加 `--ncc-win`。實際上已經有專門的腳本：

```
voxelmorph-code/scripts/torch/train/train_NCCPatchSize.py
```

exp7 / exp8 就是用它跑的。手冊寫得像還沒做，下一個 AI 可能會重複實作一遍。

**驗證指令**：
```bash
ls voxelmorph-code/scripts/torch/train/
```

---

## 3. 🔴 三份文件都沒提的東西：`fs_pipeline/`

**這是我最擔心的一項。**

`CLAUDE.md` 的「待辦 1：接入 FreeSurfer 標籤 → 改用 Dice 評估」讀起來像還沒動工。
但工作區裡**已經有寫好的程式**：

```
fs_pipeline/preprocess_fs.py         ← FreeSurfer 產物 → npz（含 seg）
fs_pipeline/verify_seg_transform.py  ← 驗證 affine 共用 + 最近鄰內插真的成立
```

我讀了兩支的檔頭 docstring，`preprocess_fs.py` 已經實作了：

- 不做 N4、不做去顱骨（FreeSurfer 的 `nu.mgz` / `brain.mgz` 已處理過），但保留 `--n4` / `--brain-extract` 可強制開啟
- aseg 用**與影像完全相同**的 Affine 變換 + 最近鄰內插搬過去
- **切分以「受試者」為單位**（避免 `A016_1` / `A016_2` 同一人橫跨 train/test 的 data leakage）
- 內建排除清單 `DEFAULT_EXCLUDE = ['A043', 'T085', 'A012', 'T065']`，且註明來源是
  `D:\MyHome\MRI\FreeSurfer\docs\ASD_資料品質記錄.md`
- `--dry-run`（只看歸戶與切分，不動影像）
- `--only <subject>`（先驗證 1 顆）
- 輸出 `split.json`（切分可重現、可複核）

`verify_seg_transform.py` 更值得一提——它的 docstring 直接寫：

> 交接文件第 5 節的程式碼作者自陳「沒在你們環境跑過」，這支腳本補上驗證。

它用合成標籤做 **bit-exact 斷言**（`quantize(img_nn)` 必須逐 voxel 等於 `seg_nn`），
還加了反證（改用 linear 內插必須出現非整數值）。不需要 FreeSurfer 資料就能跑。

**現況**：兩支都還是 untracked，也還沒有 `fs_check/` 或 `fs_preprocessed_v1/` 輸出資料夾，
代表**寫好了但還沒實際跑過**。

> 🔴 **給下一個 AI**：如果使用者叫你「接 FreeSurfer 標籤」，
> **先讀 `fs_pipeline/` 這兩支，不要從頭寫。** 該做的是跑 `verify_seg_transform.py` 驗證，
> 再用 `--only` 跑 1 顆確認，而不是重新實作一遍。
>
> 同時建議把這段補進 `CLAUDE.md` 的待辦 1，把狀態從「待辦」改成「程式已就緒，待資料」。

**驗證指令**：
```bash
ls fs_pipeline/
head -40 fs_pipeline/preprocess_fs.py
ls -d fs_check fs_preprocessed* 2>/dev/null || echo "(尚無輸出，代表沒跑過)"
```

---

## 4. `CLAUDE.md` 本身對不上的地方

### 4.1 `test.py` 不存在

`CLAUDE.md` 的目錄結構寫：

```
├── test.py       # OASIS 用（需要 seg，IXI 跑會報錯）
```

「已知問題」表格也有一列「`test.py` 讀 `atlas['seg']` 報 KeyError」。

**實際檔案叫 `test_oasis.py`。** 完整清單（`ls voxelmorph-code/scripts/torch/`）：

```
batch_test_ixi.py    batch_test_oasis.py    register.py
test_ixi.py          test_oasis.py          train/    train.py
```

手冊 §7 和實作指南 §6 也都還在寫 `test.py`，三份一起錯。

---

### 4.2 `log/exp3_script.txt` 不存在，`exp4` 的副檔名重複

`CLAUDE.md` 說「exp3–exp6 的指令在 `log/exp*_script.txt`」。實際 `ls log/`：

```
exp3.txt              ← 只有 stdout，沒有 script
exp4.txt
exp4_script.txt.txt   ← ⚠️ 副檔名重複
exp5.txt
exp5_script.txt
exp6.txt
exp6_script.txt
train_IXI.txt
train_oasis.log
```

所以實際只有 **exp4 / exp5 / exp6** 有 script log，exp3 沒有。

**建議**：把 `exp4_script.txt.txt` 改名成 `exp4_script.txt`，並修正 `CLAUDE.md` 的敘述。
（改名前先問使用者，這是他的檔案。）

---

### 4.3 `voxelmorph-code/data/` 少列了 6 個檔

`CLAUDE.md` 只寫 `data\atlas.npz`。實際有：

```
atlas.npz                    ← OASIS 原始 atlas
labels.npz                   ← ⭐ 30 個評估用標籤的 FreeSurfer ID
generated_uncond_atlas.npz
prob_atlas.npz
prob_atlas_T1_stats.npz
prob_atlas_mapping.npz
test_scan.npz
```

`labels.npz` 特別重要——**待辦 1 要改用 Dice 評估時會直接用到它**，
但三份文件裡只有實作指南 §2 提了一句，`CLAUDE.md` 完全沒有。

實作指南記錄的 30 個 ID 是：
```
[2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,31,
 41,42,43,46,47,49,50,51,52,53,54,60,63]
```
（我沒有實際載入 `labels.npz` 核對，下一個 AI 接 Dice 時請驗一次。）

---

### 4.4 我實測解掉的一個矛盾：`train_avg` 是什麼

三份文件對 `atlas.npz` 的 `train_avg` 說法不一致，我直接載入來看：

```bash
python3 -c "
import numpy as np
d=np.load('voxelmorph-code/data/atlas.npz')
for k in d.files:
    a=d[k]
    print(f'{k:12s} shape={str(a.shape):20s} dtype={a.dtype} min={a.min():.4f} max={a.max():.4f}')
"
```

結果：

```
vol          shape=(160, 192, 224)      dtype=float32 min=0.0000 max=0.7276
seg          shape=(160, 192, 224)      dtype=float32 min=0.0000 max=85.0000
train_avg    shape=(256,)               dtype=float64 min=0.0000 max=0.4186
```

**判定**：
- 手冊 §1.1 說「訓練集各標籤的平均 Dice，供 test.py 比較用」→ ✅ **正確**
  （256 正好是 FreeSurfer label ID 的範圍 0–255）
- 實作指南 §2 說「訓練集平均影像（可選用）」→ ❌ **錯的**，那是 1D 陣列不是影像

**建議**：修掉實作指南那一行。

---

### 4.5 `models/` 裡有兩個沒提到的預訓練權重

```
models/atlas_creation_uncond_NCC_1500.h5
models/vxm_dense_brain_T1_3D_mse.h5
```

看副檔名是 TensorFlow 版的官方權重。`CLAUDE.md` 只列了 exp1–exp8。

另外 `share_models/` 是**空資料夾**，三份文件都沒提，不確定用途。

---

### 4.6 `IXI/` 目錄結構的小缺漏

`CLAUDE.md` 沒列到的：`visualize_preprocess_ixi.py`、`SeeHeader.m`、`orientation_check.png`。

手冊 §7 的結構更舊——它把 `verify_orientation.py` 等三支列在 `IXI/` 根目錄，
但它們**已經移到 `IXI/verify/` 子資料夾**。手冊也缺 `NPZtoNII/`、`orientation_verify/`、`preprocess_vis/`。

---

### 4.7 git 現況可以講得更具體

`CLAUDE.md` 只說「目前工作區有未 commit 的修改」。實際 `git status --short`：

```
 M .gitignore
 M IXI/NPZtoNII/NPZtoNII.py
 M cuda_cheak.py
 M log/exp4.txt
 M log/exp5.txt
 M log/exp6.txt
 M log/exp6_script.txt
 M voxelmorph-code/voxelmorph.egg-info/PKG-INFO
 M 更動記錄.txt
?? .claude/
?? CLAUDE.md
?? FreeSurfer_到_VoxelMorph_交接.md
?? fs_pipeline/
```

值得注意：**`CLAUDE.md` 自己、`FreeSurfer_到_VoxelMorph_交接.md`、`fs_pipeline/` 全都是 untracked**。
這幾份是目前最重要的資產卻沒進版控。`.claude/` 也是 untracked，可能該決定要 commit 還是進 `.gitignore`。

> ⚠️ `CLAUDE.md` 有交代「動 git 之前先問使用者」，請遵守。

---

## 5. `VoxelMorph_PyTorch_實作指南.md` 的問題

這份最舊，整份停在 OASIS 時期。除了 §4.4 那個 `train_avg` 錯誤，還有：

| 問題 | 說明 |
|------|------|
| **整份是 Linux 語法** | `source vxm_env/bin/activate`，但這是 Windows 原生環境，應為 `.\vxm_env\Scripts\activate` |
| **§6、§7 的腳本全部改名了** | `test.py`→`test_oasis.py` / `test_ixi.py`；`batch_test.py`→`batch_test_oasis.py` / `batch_test_ixi.py`；`draw-img/visualize_registration.py`→`visualize_reg_oasis.py` / `visualize_reg_ixi.py`。§7 的輸出檔名（fig4/fig5/fig6_*.png）也對不上現在的 5 種圖 |
| **`--lambda 1.0` 建議值誤導** | train.py 預設是 0.01，實驗實際用 0.005 / 0.05，差兩個數量級 |
| **沒寫 `--image-loss` 預設是 mse 這個坑** | exp6 就是踩這個跑成 MSE。`CLAUDE.md` 有寫，指南沒有 |
| **完全沒有 IXI 這條線** | 主線早就轉到 IXI + MNI152 了，至少該加一句轉去 `CLAUDE.md` |
| **基準線數字自相矛盾** | §6.3 說未配準基準 0.6565，§6.5 說 Affine only 基準 0.584，沒解釋差別 |
| **缺「不能只看 Dice/NCC」的鐵則** | `CLAUDE.md` 有強調要看 %\|J\|≤0 的 trade-off，指南完全沒有 |
| **§4.3 `SEG35_TO_FS` 只列 7 行就 `...`** | 接下來做 aseg 標籤時，完整表會有用 |
| **Markdown 壞掉** | §6.5 和 §9 問題6 混用 ` ``` ` 和 `~~~` 導致巢狀壞掉 |
| **§8 標題打錯字** | 「模型原理y」多一個 y |
| **§6.5 有 AI 回話殘留** | 正文中間夾了一句「Created batch testing script for VoxelMorph models」 |

---

## 6. 我沒能確認、需要問使用者的事

1. **exp7 / exp8 的參數**（資料版本、λ、`--ncc-win`）
   `CLAUDE.md` 自己標了「待確認」。我只能確認 `models/exp7/` 和 `models/exp8/`
   各有 `epoch_curve.csv`、`epoch_curve.png`，和選出的 epoch 資料夾（`0155/`、`0122/`）。
   **不要自行推測填表**，問使用者。

2. **FreeSurfer 端資料是否就緒**
   `CLAUDE.md` 待辦 1 提到「開跑前先確認 FreeSurfer 端『混掃描』全面檢查已完成」。
   我看不到 `D:\MyHome\MRI\FreeSurfer`，無法確認。

3. **`share_models/` 空資料夾的用途**

4. **`.claude/` 要 commit 還是 gitignore**

---

## 7. 建議的修正順序

如果使用者說「都修吧」，我建議這個順序（危險的先修）：

1. **手冊 §6.1 的 crop ❌** — 會讓人退回 v2
2. **手冊 §4.5 / §6 的 `--target-shape`** — 給了會直接報錯的解法
3. **把 `fs_pipeline/` 寫進 `CLAUDE.md` 待辦 1** — 避免下一個 AI 重工（我認為這項實際影響最大）
4. **三份文件的 `test.py` → `test_oasis.py`** — 一起改
5. **兩份舊文件開頭加時間點警語**，並修掉 `CLAUDE.md` 裡「先讀 ixi相關手冊」那句的定位
6. `CLAUDE.md` 補 `labels.npz`、`.h5` 權重、`IXI/` 缺漏項、git 現況細節
7. 實作指南的 `train_avg` 錯誤 + Linux 語法 + 壞掉的 Markdown
8. `log/exp4_script.txt.txt` 改名（要先問）

---

## 8. 給下一個 AI 的提醒

- **`log/*.txt` 和 `requirements.txt` 是 UTF-16LE**，直接 `cat` 會是亂碼。
  用 `iconv -f UTF-16 -t UTF-8` 或 `open(..., encoding='utf-16')`。（這條 `CLAUDE.md` 有寫，很有用）
- **`models/exp1` 有 405 個 `.pt`、`exp4` 有 504 個，不要遍歷讀取**，只讀 `epoch_curve.csv`。
- 這是 **Windows 原生環境**，路徑用 `\`，PowerShell 換行用反引號。
- **本專案不含 TransMorph。** TransMorph 在 `D:\MyHome\MRI\TransMorph\`，有自己的 `CLAUDE.md`。
  根目錄那份 `TransMorph_Report.docx` 只是報告備份，跟這裡的程式碼無關。
- 我這次**只讀沒改**。如果你發現有檔案被改動，那不是我做的——
  上面 §4.7 列的那些 modified 檔案在我進來之前就是那個狀態了。

---

*以上。祝你修得順利。*
