# ASD 資料集：FreeSurfer 產物 → VoxelMorph 訓練實作手冊

何承運 · 2026
建立日期：**2026-08-23**　·　最後更新：**2026-09-13**

---

> ## 這份文件的定位
>
> 記錄**老師提供的 ASD 資料**從 FreeSurfer 產物接進 VoxelMorph 的完整過程。
> 對照 `IXI/ixi相關手冊.md`（那份是 IXI 那條線）。
>
> - **專案總覽與最新狀態**：`CLAUDE.md`
> - **FreeSurfer 端寫給我們的接入說明**：`FreeSurfer_到_VoxelMorph_交接.md`
> - **FreeSurfer 端的資料品質原始記錄**：`D:\MyHome\MRI\FreeSurfer\docs\ASD_資料品質記錄.md`
>
> ## 現況（2026-09-13）
>
> | 階段 | 狀態 |
> |---|---|
> | 資料把關 | ✅ DICOM 檔頭逐一核對：ASD 170 → **164**；DGM 54 個掃描 = 52 人；VNT 68 人（§15）|
> | 前處理 | ✅ 三包 FreeSurfer（train 258 / test 28）＋ tigerbx 一包（同一個切分），0 失敗 |
> | 訓練 | ✅ mix_exp1（FreeSurfer 標籤）、tiger_exp1（tigerbx 標籤），都在 B 台 |
> | Dice 評估 | ✅ mix 0.7874、tiger 0.8594，折疊率皆 0（§16、§17）|
> | 跟論文比 | ✅ Table I 全表、逐項差異、推論時間實測（§18）|
> | meeting 簡報 | ✅ `meeting報告/ASD延伸實驗_混合訓練與tigerbx_v2.pptx`（29 頁）|
>
> ⚠️ §2–§13 寫於 08-23～09-07（167 顆清單、asd_exp1 時期），保留當歷史。
> 跟 §15 以後衝突時，**以 §15 以後為準**。

---

## 1. 資料來源

| 項目 | 內容 |
|---|---|
| 提供者 | 老師（研究用 ASD 資料） |
| 掃描儀 | Siemens Skyra 3T |
| 序列 | MPRAGE，1mm |
| 原始資料位置 | Ubuntu VM：`/mnt/hgfs/outside/ASD/`（.IMA） |
| FreeSurfer 輸出 | Ubuntu VM：`/home/cheng/workspace/subjects/ASD_result` |
| 資料夾總數 | **170** |
| `recon-all` 跑完 | **167**（9 天 17 小時），0 失敗 |
| **最終可用（FINAL）** | **167** ★ 混掃描檢查通過，異常 0 |

⚠️ **這批不是 IXI。** IXI 是健康成人（20–86 歲、三台機器），ASD 這批是另一個 cohort。
兩者混用或跨批比較的取捨見 §8。

---

## 2. 🔴 資料品質：排除清單

> 🔄 **2026-09-07 更新**：這張表是 08-23 版。之後讀了 DICOM 檔頭，T065 翻案納回，
> 另外排除 A016_1 / A016_2 / YT13 / A0132，清單變成 164。**最新的排除表見 §15.1。**

來源：`D:\MyHome\MRI\FreeSurfer\docs\ASD_資料品質記錄.md`（FreeSurfer 端已查證，有 log／檔案證據）

| 受試者 | 問題 | 處置 |
|---|---|---|
| **A043** | 影像雜訊過高、灰白對比不足（白質只認出約 5%）| 🔴 排除（recon-all 未完成）|
| **T085** | 只有 120/192 張切片（缺 0001–0072）+ 首張檔案截斷 | 🔴 排除（來源即缺）|
| **A012** | 資料夾混了兩次掃描（0801 + A012 各 192 張 → 合併成 384）| ✅ **2026-08-23 已修復並重跑完成（141 分），已納入 FINAL 清單** |
| **T065** | 資料夾名 T065，但 DICOM 病人 ID 是 **T056** | 🟠 身分待確認，先不納入 |

> ⚠️ **A012 這一列是「狀態會變」的活教材。**
> `preprocess_fs.py` 的 `DEFAULT_EXCLUDE` 原本寫死了這四個，A012 修好之後，
> 那份內建表會把 FINAL 清單的 167 **安靜地砍成 166**——不會有任何錯誤訊息。
> 已改成：**有 `--subject-list` 時，清單才是權威**，內建表只提示不砍人
> （會印出「清單裡有 N 個曾被判定有問題的受試者，依清單為準予以保留」）。
> 要覆寫請明寫 `--exclude`。

`A043` / `T085` / `T065` 已從來源資料夾移除，本來就不會出現在 `--img-dir` 裡。

### 混掃描全面掃描

**A012 是因為疊成 384 層超過 FreeSurfer 的 256 上限才報錯露餡。**
如果某個資料夾也混了兩次掃描、但總層數剛好塞得進 256，**`recon-all` 不會報錯**，
會安靜跑出一顆「兩個人疊在一起」的腦。這種資料餵進訓練會讓模型學到不存在的解剖結構。

### ✅ 狀態：2026-08-23 已完成，通過

| 項目 | 結果 |
|---|---|
| 掃描資料夾 | 167 個 |
| 異常 | **0** |
| 每個資料夾的 .IMA 張數 | 全部恰好 **192** |
| 病人 ID | 每個資料夾只含單一 ID |
| 數字閉合 | 167 × 192 = **32,064**，與 `find` 掃到的檔案總數完全一致 |

同時反證了三件事：A012 的修復確實成功（384 → 192）、**沒有其他資料夾混掃描**、沒有缺切片的。

> ⚠️ **殘餘風險（FreeSurfer 端誠實告知）**：本次驗證的是「檔案數 + 病人 ID 一致性」，
> **沒有逐檔驗證檔案完整性**（例如某張被截斷）。但 167 顆全部通過 `recon-all -all`
> （需要能完整讀取整個序列才可能成功），這是足夠強的旁證。

> 📌 執行細節：第一版指令在 VMware 共享資料夾上跑爆（EAGAIN "Try again"，
> 對 167 個資料夾各跑多次 `ls`，HGFS 撐不住）。改成「`find` 一次性抓清單存檔 →
> 純本機分析」才跑完。之後在 HGFS 上做大量檔案操作要注意這點。

🔓 **閘門條件已滿足**，可加 `--list-is-final` 批次執行。

---

## 3. 受試者歸戶（避免 data leakage）

> 🔄 **2026-09-07 已解決**：DICOM 檔頭顯示 A013 是另一個人，A0131 / A0132 是同一人（A0132 排除），
> A016_1 / A016_2 都排除。本節的「暫定假設」已經不需要。DGM 另有兩對同一人，用 `ASD/DGM_groups.txt`。見 §15。

同一個人的多次掃描若一個進 train、一個進 test，模型等於看過答案，Dice 會虛高。
**切分必須以「受試者」為單位，不是檔案層級 shuffle。**

### 🔴 本次採用的假設：完全不合併（`--grouping none`）

**使用者決定（2026-08-23）**：

| 組別 | 檔名 | 決定 |
|---|---|---|
| A013 | `A013`, `A0131`, `A0132` | **當作三個不同的人** |
| A016 | `A016_1`, `A016_2` | **當作兩個不同的人** |
| 0801 | — | 不納入（修 A012 時已移到 `ASD_T1only_misplaced/`，本來就不在 167 裡）|

**→ 167 個掃描 = 167 位受試者，不做任何合併。**

⚠️ 這跟程式的預設行為（`--grouping auto` 會自動合併 `_1`/`_2`）相反，
**所以每次都要明寫 `--grouping none`**，否則會變成 166 位。

### ⚠️ 這是暫定假設，未經老師確認

若 A013 / A016 那幾組其實是同一人的多次掃描，而被分到不同的 train/test，
會造成 **data leakage，Dice 會虛高**。

FreeSurfer 端粗估：在 10% test 比例下，A013 那組約 **27%**、A016 那組約 **18%**
的機率會被拆開。影響範圍有限（最多 1–3 個 test 樣本），但：

> 🔴 **若要發表，這個假設必須在方法學中說明，或先向老師確認。**

程式每次執行都會把這兩組列出來並印出風險警告，不會安靜通過。

### 好消息：可逆，不用改程式

歸戶是**外部設定檔**。老師之後若確認為同一人，寫一個 `ASD/groups.txt`（TSV）：

```
A013	A013
A0131	A013
A0132	A013
A016_1	A016
A016_2	A016
```

然後加 `--group-map ASD\groups.txt` 重跑切分即可。`--group-map` 明列的一律優先於 `--grouping`。

📌 **`split.json` 會記錄本次採用的 `grouping` 模式與被拆開的組別**，
之後回頭看才知道那份切分是在什麼假設下產生的。

---

## 4. 從 FreeSurfer 端轉檔過來

### 4.1 在 Ubuntu VM 上執行（FreeSurfer 環境）

```bash
export SUBJECTS_DIR=/home/cheng/workspace/subjects/ASD_result
OUT=/mnt/hgfs/outside/fs_for_vxm        # ← 寫到共享資料夾
mkdir -p $OUT/norm $OUT/aseg

for d in $SUBJECTS_DIR/*/; do
  s=$(basename "$d")
  grep -q "finished without error" "$d/scripts/recon-all.log" 2>/dev/null || continue
  mri_convert "$d/mri/norm.mgz" "$OUT/norm/${s}.nii.gz"
  mri_convert "$d/mri/aseg.mgz" "$OUT/aseg/${s}.nii.gz"  -rt nearest
done
```

- 只轉「跑成功」的受試者
- **影像一律用 `norm.mgz`**（見下方「為什麼是 norm 不是 brain」），全體一致，沒有混用
- 實際產出：**167 顆，norm/ + aseg/ 合計 285 MB**

### 🔻 為什麼是 `norm.mgz` 不是 `brain.mgz`（2026-08-23 變更）

| 檔案 | 怎麼來的 |
|---|---|
| `norm.mgz` | `mri_ca_normalize` 的產物：用 **GCA 圖譜**當控制點做亮度正規化，再遮罩 |
| `brain.mgz` | 再多做一次 `mri_normalize -aseg`，改用**該受試者自己的 aseg** 當控制點 |

也就是 `brain` 比 `norm` 多一道、而且更貼身的正規化。

FreeSurfer 端原本推薦 `brain`（跨受試者亮度更一致）；**使用者選 `norm`**
（在 VoxelMorph 生態裡更常見，理由是少一道處理、離原始更近）。

> **兩者都是合理選擇，目前沒有文獻依據說哪個對配準訓練更好。**
> 重點是**全體一致**——這點有做到，167 顆全是 `norm.mgz`。
> 📌 要發表的話，方法學要寫明用的是 `norm.mgz`。

**對前處理的影響：零。** `norm.mgz` 是 uchar（0–255），FreeSurfer 已把白質峰值定錨在 110，
跨受試者一致。第 5 步的 percentile(1,99) clip + min-max 照樣適用，不需要改參數
（其實因為已經正規化過而近乎冗餘，但保留它可以跟 IXI 流程維持一致，**不要拿掉**）。
`aseg.mgz` 完全沒變，已驗證的標籤搬運邏輯不受影響。

### 4.2 搬到 Windows

> 🔴 **重要：資料在「另一台電腦」上。**
> `/mnt/hgfs/outside` 是**跑 FreeSurfer VM 的那台機器**的 VMware 共享資料夾，
> 不是本專案這台。在本機的 `D:` 和 `C:\Users` 都找不到該路徑。
> 所以那 285 MB 需要**跨機器搬運**（隨身碟／區網／雲端都可以，285 MB 不大）。

> ⚠️ **待填**：搬到本機後的路徑 = `______________`

⚠️ **VMware 共享資料夾對大量檔案複製不太可靠**（T085 最初就被懷疑是複製出問題，
後來確認是來源本身缺檔）；而且混掃描檢查時第一版指令就在 HGFS 上跑爆過（EAGAIN）。
搬完務必核對檔案數與大小。
`preprocess_fs.py` 會列出「白名單裡有但 `--img-dir` 找不到」以及「缺對應 aseg」的 ID，
可以當作核對工具。

### 4.3 放到哪

```
ASD/
├── norm/                ← norm.mgz 轉出的 .nii.gz（167 個）
├── aseg/                ← aseg.mgz 轉出的 .nii.gz（167 個）
├── subjects_final.txt   ← ✅ 已就位（FINAL，167 個 ID）
└── groups.txt           ← 只在老師確認 A013/A016 是同一人時才需要
```

> 📌 `.gitignore` 已設定：`ASD/` 底下的**資料檔會被忽略**，但 `*.py` / `*.md` /
> **根目錄的 `*.txt`**（清單、歸戶表）會進版控。清單是重要溯源資料，不要放進子資料夾。

---

## 5. 前處理：`preprocess_fs.py`

### 5.1 和 IXI 的差別

| 步驟 | IXI (`preprocess_ixi.py`) | ASD (`preprocess_fs.py`) |
|---|---|---|
| N4 bias correction | ✅ 做 | ❌ **跳過**（FreeSurfer 的 `nu.mgz` 階段已做）|
| 去顱骨 | ✅ antspynet（最慢的一步）| ❌ **跳過**（`norm.mgz` 已去過）|
| Affine 對到 atlas | ✅ | ✅ **一樣，沿用同一個 atlas** |
| 搬 aseg 標籤 | — | ✅ **新增**（同一變換 + 最近鄰）|
| 正規化 [0,1] | ✅ | ✅ 一樣 |
| 切分 | 檔案層級 shuffle | **受試者層級** |
| npz keys | `vol` | `vol` + **`seg`** |

> `--n4` / `--brain-extract` 可以強制開啟，預設關閉。

⚠️ **不要用 FreeSurfer 的 `talairach.xfm`** —— 它對到 **MNI305**，我們的 atlas 是 **MNI152 09c**，
不是同一個空間。繼續用 `ants.registration()` 對到我們自己的 atlas，這樣 shape (192,224,192)、
正規化方式、既有 IXI 資料全部相容。

### 5.2 🔒 執行閘門

**沒有 `--list-is-final` 旗標時，腳本拒絕批次執行**（exit code 2），
只允許 `--dry-run` 和 `--only <SUBJECT>`。

🔓 **2026-08-23：閘門條件已滿足。** 混掃描檢查通過（異常 0），FreeSurfer 端已交付
標明 FINAL 的清單（167）並明確授權解除。加 `--list-is-final` 即可批次執行。

> 📌 這個閘門的設計目的是「不靠記憶」：資料就緒與否由旗標決定，不是靠人記得有沒有查過。
> 之後若換資料集，同樣的機制可以再用一次。

### 5.3 執行順序

> ✅ **這一節已經執行完畢（2026-08-23）**，輸出在 `data/ASD_preprocessed_v1/`。
> 以下保留完整步驟，供之後換資料或需要重跑時參考。

**最省事的做法是用包裝腳本**，它會自動帶入正確參數（`--grouping none`、
`--list-is-final`）、跑起跑前檢查、顯示切分讓你確認、跑完抽驗 3 顆：

```
python ASD\run_preprocess.py --src-dir data\ASD_data --out-dir data\ASD_preprocessed_v1 --dry-run   # 只看切分
python ASD\run_preprocess.py --src-dir data\ASD_data --out-dir data\ASD_preprocessed_v1             # 正式跑（會問你確認）
```

底下是它實際呼叫的指令，需要自訂參數時可以直接用。

> 清單來源：`D:\MyHome\MRI\FreeSurfer\docs\ASD_可用清單_FINAL.txt`（167 個，檔頭有 `#` 註解，
> 程式會自動跳過）。已複製一份到 `ASD/subjects_final.txt` 進版控。
> ⚠️ 舊的 `ASD_可用清單_暫定.txt`（166）**已被 FreeSurfer 端刪除**，避免誤用。

**① 先看歸戶與切分（不動影像）**
```powershell
python ASD\preprocess_fs.py `
    --img-dir data\ASD_data\norm --seg-dir data\ASD_data\aseg `
    --atlas IXI\atlas_mni152_09c_v3.nii.gz `
    --out-dir data\ASD_preprocessed_v1 `
    --subject-list ASD\subjects_final.txt `
    --grouping none `
    --dry-run
```

用 FINAL 清單（167）實跑的結果：

```
歸戶：167 個掃描 -> 167 位受試者
  --grouping  : none（每個掃描各自成一人，不合併）
[!] 以下 2 組（共 5 個掃描）疑似同一人，但目前被當成不同人：
      ['A013', 'A0131', 'A0132']
      ['A016_1', 'A016_2']
切分（受試者層級，seed=42，test_frac=0.1）：
  train  150 人 / 150 個掃描        ← 切分當下；A016_1 之後於 QC 移出，實際訓練 149
  test    17 人 /  17 個掃描
  [v] 沒有受試者橫跨 train/test
```

**② 驗證 1 顆**（產生 nii 供 ITK-SNAP / Freeview 目視確認）
```powershell
python ASD\preprocess_fs.py `
    --img-dir data\ASD_data\norm --seg-dir data\ASD_data\aseg `
    --atlas IXI\atlas_mni152_09c_v3.nii.gz `
    --out-dir ASD\fs_check --only A001 --save-nii
```
目視要確認三件事：
1. 影像有對到 atlas 空間、shape = (192, 224, 192)
2. **標籤和影像完全疊合**（最重要）
3. 標籤值仍是整數，沒有出現奇怪的中間值

**③ 批次跑（閘門已解除）**
```powershell
python ASD\preprocess_fs.py `
    --img-dir data\ASD_data\norm --seg-dir data\ASD_data\aseg `
    --atlas IXI\atlas_mni152_09c_v3.nii.gz `
    --out-dir data\ASD_preprocessed_v1 `
    --subject-list ASD\subjects_final.txt --list-is-final `
    --grouping none `
    --save-nii > .\log\asd_preprocess.txt 2>&1
```

⚠️ **不要加 `--group-map`**，除非老師已確認 A013 / A016 那幾組是同一人（見 §3）。

### 5.4 輸出

```
data/ASD_preprocessed_v1/
├── train/<subject>.npz    keys: vol (float32 [0,1]) + seg (int16)
├── test/<subject>.npz
├── nii/                   （--save-nii：<subj>.nii.gz + <subj>_seg.nii.gz）
└── split.json             ← 切分結果，可重現、可複核
```

---

## 6. ✅ 已驗證的事實

### 6.1 標籤搬運（`verify_seg_transform.py`）

交接文件第 5 節的程式碼原本標註「未在本環境實測」。已補上驗證：

**測試設計**：用同一個 `reg['fwdtransforms']` 把**影像**和**標籤**都以 nearestNeighbor 搬過去，
再把搬完的影像用原門檻重新量化，與搬完的標籤逐 voxel 比對。最近鄰對兩者取的是同一顆
來源 voxel，所以只要變換真的共用，兩者必然 bit-exact。

**結果（ANTsPy 0.6.3，真實 IXI 影像 + 合成標籤）**：

| 測試 | 結果 |
|---|---|
| 標籤 shape = atlas | ✅ (192,224,192) |
| 搬完仍是整數 | ✅ 與最近整數最大偏差 = **0** |
| 沒產生不存在的標籤值 | ✅ [0,10,17,41] → [0,10,17,41] |
| **影像與標籤逐 voxel 對齊** | ✅ **不一致 0 / 8,257,536** |
| 反證：改用 `linear` | ✅ 捏造出 **1,536,912** 個原標籤集裡沒有的值 |
| 體積變化合理 | ✅ 比值 0.690（Affine 縮放，正常）|

重跑方式：
```powershell
python ASD\verify_seg_transform.py `
    --img IXI\IXI-T1\IXI002-Guys-0828-T1.nii.gz `
    --atlas IXI\atlas_mni152_09c_v3.nii.gz
```

### 6.2 訓練端相容性

- npz 有 `vol` + `seg` 兩個 key 時，`voxelmorph/py/utils.py:63` 走 `npz[np_var]`
  （`np_var` 預設 `'vol'`）→ **正確取到 `vol`，不會拿到 `seg`**
- `train.py` 用作者預設值實跑 2 epoch，正常收斂存檔
- **VoxelMorph 那邊完全不用改**

### 6.3 標籤三鐵則（已寫成程式裡的硬檢查）

1. 必須用**與影像完全相同**的變換 —— 不能各自對位一次
2. 內插必須用**最近鄰** —— `linear` 會在標籤 17 和 10 之間插出 13.5 這種不存在的值
3. 標籤**不正規化、不轉 float** —— 保持 `int16`

> 搬完若出現非整數值，`preprocess_fs.py` 直接 `raise`，不會安靜存進 npz。

---

## 7. 訓練（在 B 台跑）

### 7.1 兩台機器的分工

| | A 台（主機） | B 台（訓練機） |
|---|---|---|
| 路徑 | `C:\Users\h4524\claude_cheng` | `D:\chengyun\TRY_voxelmorph` |
| 負責 | 前處理、評估、視覺化、文件 | **只跑訓練** |
| 需要 ANTs / antspynet | ✅ | ❌ **不用裝** |
| 需要原始 `.IMA` / `norm.mgz` | ✅ | ❌ **不要搬** |

**為什麼這樣切**：前處理只需要做一次，而且 `antspynet` 相依 TensorFlow，
在 Windows 上很麻煩（原生 TF ≥ 2.11 還沒有 GPU）。B 台只要 PyTorch 就夠了。

📌 **順帶的好處**：npz 裡只有 `vol` 和 `seg` 兩個 numpy 陣列，
**沒有 DICOM header、沒有受試者姓名**，檔名也只是 `A001` 這種代碼。
搬 npz 到 B 台等於順便去識別化。**原始 `.IMA` 的檔名與 DICOM 檔頭有真名，不要搬、也不要進 git。**

### 7.2 要搬什麼

程式和文件走 GitHub（`git pull` 就有）。**只有兩樣要另外搬**，因為 `.gitignore` 擋著：

| 從 A 台 | 大小 |
|---|---|
| `data\ASD_preprocessed_v1\` | **1.28 GB** |
| `IXI\atlas_mni152_09c_v3.npz` | 6 MB |

`data/ASD_preprocessed_v1/MANIFEST.json` 有 167 個檔案的名稱與大小，搬完可以核對。

⚠️ **atlas 版本必須跟前處理用的一致**（都是 v3）。混用會報
`Sizes of tensors must match`，或更糟——安靜地訓練出對不準的模型。

### 7.3 在 B 台怎麼跑

```
python ASD\run_train.py --train-dir data\ASD_preprocessed_v1\train --exp-name asd_exp1 --check-only   # 先檢查，不訓練
python ASD\run_train.py --train-dir data\ASD_preprocessed_v1\train --exp-name asd_exp1                # 正式跑
python ASD\run_train.py --train-dir data\ASD_preprocessed_v1\train --exp-name asd_exp1 --resume       # 中斷後續跑
```

`--check-only` 要確認的三行：**python 路徑是不是你的 venv**、
**`ASD_preprocessed_v1\train` 有沒有 149 個 npz**、**GPU 有沒有抓到**。

> 📌 包裝腳本刻意寫成 **Python 而不是 `.ps1`**：
> cmd.exe 不會執行 `.ps1`（只會用預設關聯程式開啟，看起來像「跳出編輯器」），
> PowerShell 還要處理 ExecutionPolicy 與 BOM。
> Python 版在 cmd / PowerShell 都能跑，而且用 `sys.executable`，
> **就是你當下啟動的那個 venv，不可能抓錯**。

`run_train.py` 會自動把指令存成 `log/asd_expN_script.txt`（含機器名稱與資料版本）。

### 7.4 超參數

🔴 **`--image-loss` 預設是 `mse`**，想跑 NCC 一定要明寫。

🔴 **λ 的尺度取決於 `--image-loss`**：論文對 CC 的最佳值是 **≈1–2**，對 MSE 是 0.01–0.02。
`train.py` 預設的 0.01 是為 MSE 設的，**搭 NCC 用等於幾乎沒有正則化**。
IXI 那邊的 exp3–exp8 就是踩了這個（平滑項只佔損失 1–3%）。詳見 `CLAUDE.md`「超參數」節。

**asd_exp1 採用 `ncc` + `λ=1.0`**（使用者決定）。這是本專案第一次把 λ 放到論文建議的尺度。

### 7.5 ⚠️ 沒有 validation set

**現況：train 149 / test 17，沒有 val。**

`train.py` 本身不做任何評估；挑 epoch 是靠 `batch_test_ixi.py` 掃過所有 `.pt`，
在 **test set** 上算指標再取綜合分數最高的（`batch_test_ixi.py:204`）。

> 🔴 **這代表「挑 epoch」和「報成績」用的是同一批資料**，test 實質上變成 validation，
> 報出來的數字會樂觀偏高。
>
> 論文（TMI 2019 §V-A）的做法是 3231 train / **250 validation** / 250 test，
> 「select the network that optimizes Dice on our validation set, and report results on our test set」。

**使用者決定維持 149/17**（跟 IXI 那條線 exp4–exp8 的現行做法一致）。
📌 **要發表的話，這點必須在方法學中明講。**

改的話很便宜：從 `train/` 挪 17 顆出來即可（133/17/17），**不用重跑前處理**——
npz 已經產生好了，只是移動檔案。但訓練資料會從 149 掉到 132。

### 7.6 產出與搬回

| 產出 | 大小 | 要搬回 A 台嗎 |
|---|---|---|
| `models\asd_exp1\0000.pt` … `0250.pt` | 1.16 MB × 251 ≈ **292 MB** | ❌ 只搬最佳的 1~3 個 |
| `epoch_curve.csv` / `.png` | 350 KB | ✅ |
| `log\asd_exp1.txt` | ~2 MB | ✅ |
| `log\asd_exp1_script.txt` | < 1 KB | ✅ |

⚠️ `train.py` **每個 epoch 都存一個 `.pt`，不會自動清理**。
IXI 的 `exp4` 就留下 501 個檔、970 MB。挑完 epoch 記得清。

---

## 8. ⚠️ 研究設計：Dice 要報在哪批資料上（尚未定案）

**這題的決定權在使用者與老師，以下只是取捨依據。**

### 核心問題

IXI 訓練出來的模型拿去 ASD 資料上報 Dice，中間跨了 **cohort 和年齡**的 domain shift。

更尖銳的是**標籤方法本身就是 confound**：
若 IXI 用 SynthSeg 標籤、ASD 用 aseg 標籤，兩邊的 Dice **不能直接比較**——
差異裡混雜了「配準品質差異」和「兩套分割方法的差異」，分不開。

> **規則：在哪一批算 Dice，那批就要用同一套標籤方法。**
> 要跨批比較，兩批都跑 SynthSeg（IXI 跑 aseg 需要 581 × 2 小時 ≈ 48 天，不可行）。

### FreeSurfer 端建議的兩段式

| 段 | 資料 | 標籤 | 角色 |
|---|---|---|---|
| 第一段 | IXI（581）| 統一用 SynthSeg | **主結果**，cohort 一致，好解讀 |
| 第二段 | ASD（167）| aseg | **external validation**，明講跨 cohort / 跨機器 |

好處：保住既有的 581 顆 IXI 和訓練好的模型；ASD 樣本數少，拿去當獨立驗證剛好。
而且如果表現掉了，那本身就是**泛化能力的證據**，是有價值的結論而不是瑕疵。

### 另一個選項

ASD 這批自己 train + test，Dice 內部一致。缺點是樣本數少（167 → train 149 / test 17），
且犧牲了既有的 IXI 主結果。

---

## 9. 實驗記錄

`run_train.py` 會自動把指令存到 `log/asd_expN_script.txt`（含機器名稱與資料版本）。

| 實驗 | 資料 | atlas | epochs | image-loss | λ | int-steps | int-downsize | 最佳 epoch | Dice | %\|J\|≤0 | 備註 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **asd_exp1** | ASD_preprocessed_v1 | v3 | 250 | **ncc** | **1.0** | 7 | 2 | 0190 | **0.7811** | **0.0000%** | 已跑完（B 台）。基準線 0.6760，+0.105 |
| **mix_exp1** | mixed_preprocessed_v1（三包）| v3 | 250 | ncc | 1.0 | 7 | 2 | 0230 | **0.7874** | **0.0000%** | 2026-09-08，B 台。基準線 0.6753，+0.112（§16）|
| **tiger_exp1** | tigerbx_preprocessed_v1 | v3（seg 用 tigerbx 版）| 250 | ncc | 1.0 | 7 | 2 | 0240 | **0.8594** | **0.0000%** | 2026-09-09，B 台。基準線 0.7376，+0.122（§17）|

📌 **λ=1.0 是這個專案第一次這樣跑，而且確實生效了。**
`log/asd_exp1.txt` 最後一步：`loss: -0.169698 (-0.199097, 0.029399)`，
**平滑佔比 12.9%** —— IXI 那邊的 NCC 實驗（exp4/5/7）只有 1–3%，形同沒有正則化。

### 9.1 asd_exp1 的結果

| 項目 | 值 | 說明 |
|---|---|---|
| 基準線（只有 Affine） | **0.676014** | 模型無貢獻時的 Dice |
| 最佳 epoch | 0190 | **但這是雜訊**，見下 |
| Dice | **0.7811** | 比基準線 +0.105 |
| %\|J\|≤0 | **0.0000%** | 全部 26 個檢查點皆為零 |
| 平滑項佔比 | 12.9% | 最後一步的損失拆項 |

🟢 **評估鏈的 sanity check 通過**：`epoch 0` 的 Dice 等於 `--baseline` 算出的
0.676014。兩者走**完全不同的程式路徑**（一個載模型跑推論、一個直接搬標籤），
數字一致代表標籤搬運與 Dice 計算沒有錯。

🔴 **「最佳 epoch = 190」是雜訊，不要當成結論。**
17 顆測試的標準誤是 **0.0038**（舊版寫 0.0051，2026-09 重算更正），epoch 90 之後各檢查點差距很小。
`dice_curve.csv` 尾段：230 → 0.780848、240 → 0.780899、250 → 0.779413。

🔴 **更正（2026-09-13）：舊版這裡寫「折疊率 0 代表折疊預算沒用到、λ=1.0 可能偏保守」，這個推論不成立。**
我們用的是 repo 預設的**微分同胚版**（`int_steps=7`，速度場積分），這個版本本來就幾乎不會折疊；
論文 Table I 的 0.366% 是**非微分同胚的位移場版本**（`int_steps=0`）。兩者不能比，
折疊率 0 也不能拿來說 λ 太大或模型比較好。λ=0.5 對照仍然可以做，但目的只是「看 Dice 會不會上升」。

⚠️ **不要拿 0.7811 直接跟論文的 0.753 比。** 基準線就不一樣（我們 0.676、論文 0.584），
資料集、標籤來源、atlas 也全都不同。要比的是「比基準線高多少」。

**評估提醒**：
- Dice 用 `voxelmorph-code/data/labels.npz` 的 **30 個 FreeSurfer 標籤 ID**
  （已實際載入確認：`int64`，值為
  `[2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,31,41,42,43,46,47,49,50,51,52,53,54,60,63]`），
  正好對應論文 Table I 的 30 個結構
- **只看 Dice 會被「亂折疊硬湊高分」騙，必須同時報 %|J|≤0**
- %|J|≤0 的合理範圍：論文 VoxelMorph(CC) 是 **0.366%**、ANTs SyN 0.185%、NiftyReg 0.793%。
  **0.1% 量級屬正常**，該避開的是 2% 以上

---

## 10. 待辦 / 待確認

### ✅ 已完成

- [x] **混掃描全面檢查** —— 2026-08-23，167 個資料夾異常 0，FINAL 清單（167）已交付
- [x] **資料搬到本機** —— `data/ASD_data/norm` + `aseg`，各 167 個，285 MB，
      gzip 完整性 334/334 通過
- [x] **批次前處理** —— train 150 / test 17，0 失敗、0 個標籤消失（A016_1 之後於 QC 移出 → 149）
- [x] **單顆端到端驗證** —— A001 量化檢查 10/10 通過（含左右翻轉檢查）

- [x] **atlas 的 aseg** —— 補零 → recon-all → 兩段整數切片，5/5 驗證通過（§12）
- [x] **asd_exp1 訓練完成** —— 250 epochs，Dice 0.7811，折疊率 0.0000%（§9.1）
- [x] **Dice 評估腳本** —— `ASD/test_dice.py`（§13.1）
- [x] **視覺化** —— `ASD/visualize_dice.py`（§13.2）

### ✅ 2026-09 完成

- [x] DICOM 檔頭核對、同一人判定（§15）
- [x] DGM、VNT 前處理；三包混合（§16）
- [x] mix_exp1、tiger_exp1 訓練與 Dice 評估（§16、§17）
- [x] 跟論文 Table I 逐項對照、推論時間實測（§18）

### 🔴 下一步

- [ ] **切出 validation set** —— 使用者決定「等下次資料多一點再做」。目前 epoch 是看 test 挑的，數字偏樂觀
- [ ] （可選）λ = 0.5 對照：只看 Dice 會不會上升（**不是**因為折疊率 0，見 §9.1 更正）
- [ ] 訓練可縮成 120 epochs：epoch 90 之後已持平（§16）
- [ ] 頭殼沒切好的影響：沒有直接測試（使用者決定暫不做）。要測就把腦遮罩刻意膨脹／侵蝕，其他條件不變

### 🟠 需要問老師

- [ ] **DGM 2023-04-25 那四位**：資料夾編號與 DICOM ID 差 1
- [ ] **A014 的組別**：DICOM ID 是 T094
- ~~A013 / A016 是否同一人~~、~~T065 身分~~ → 已由 DICOM 解決（§15）

### 🟡 研究設計

- [ ] **Dice 報在哪批**（§8）—— 目前實際做法是 ASD + DGM + VNT 自己 train / test

---

## 11. 相關檔案

| 檔案 | 用途 |
|---|---|
| `ASD/preprocess_fs.py` | FreeSurfer 產物 → npz（含 seg）|
| `ASD/verify_seg_transform.py` | 驗證 affine 共用 + 最近鄰內插 |
| `ASD/verify_one_subject.py` | 單顆 10 項量化驗證（含左右翻轉檢查）|
| `ASD/make_padded_atlas_input.py` | atlas 補零成 256³，讓 recon-all 零內插（§12.1）|
| `ASD/verify_conform_roundtrip.py` | 量 conform 來回的位移（§12.2）|
| `ASD/make_atlas_seg.py` | atlas aseg 切回訓練空間（§12.3）|
| `ASD/test_dice.py` | ⭐ Dice 評估（§13.1）|
| `ASD/visualize_dice.py` | ⭐ 標籤重疊 / 輪廓 / 逐結構長條圖（§13.2）|
| `ASD/run_preprocess.py` / `run_train.py` | 前處理與訓練的包裝腳本 |
| `ASD/make_mixed_set.py` | 多資料集併成一份（預設實體複製）|
| `ASD/check_dataset.py` | 搬機器後的資料完整性檢查（sha256 manifest）|
| `ASD/find_duplicate_scans.py` | atlas 空間的標籤 Dice 找重複掃描（§15.2）|
| `ASD/DGM_groups.txt` | DGM 歸戶表（已去識別化）|
| `IXI/atlas_mni152_09c_v3_seg_tigerbx.npz` | tigerbx arm 的 atlas 分割（§17）|
| `share_models/{ASD,mix_exp1,tiger_exp1}_good/` | 進版控的最佳模型與 Dice 曲線 |
| `meeting報告/ASD延伸實驗_混合訓練與tigerbx_v2.pptx` | 2026-09 meeting 簡報（29 頁，含備忘稿）|
| `ASD/slides_src/2026-09_mix_tigerbx/` | 上面那份 29 頁簡報的產生器（pptxgenjs）|
| `ASD/slides_src/` | 舊 meeting 簡報原始碼（25 頁 `.dc.html` + `canvas.json`）|
| `meeting報告/講稿.md` | 舊簡報（25 頁）的講稿 |
| `ASD/subjects_final.txt` | 🟡 舊清單（08-23 版）；現行清單是 `data/ASD_data/fs_stats/subjects.txt`（164，隨資料附）|
| `CLAUDE.md` | 專案總覽、超參數、實驗記錄 |
| `FreeSurfer_到_VoxelMorph_交接.md` | FreeSurfer 端寫的接入說明 |
| `D:\MyHome\MRI\FreeSurfer\docs\ASD_資料品質記錄.md` | 品質問題的原始診斷 |
| `D:\MyHome\MRI\FreeSurfer\docs\ASD_可用清單_FINAL.txt` | ★ **最終清單（167）**，混掃描檢查已通過 |
| `D:\MyHome\MRI\FreeSurfer\docs\ASD_全部資料夾清單.txt` | 全部 170 個，含壞掉的 |


---

## 12. atlas 的 aseg：自己跑 FreeSurfer

**為什麼要自己跑**：Dice 要拿受試者的標籤跟 **atlas 自己的標籤**比，
但我們的 MNI152 atlas 是自製的，只有影像沒有標籤。

找過的現成來源，全都不能用：

| 來源 | 內容 | 為什麼不能用 |
|---|---|---|
| MNI 官方發布包 | gm/wm/csf 組織機率圖 + 三張二值遮罩 | **沒有解剖結構標籤** |
| CerebrA | Mindboggle-101 配準 + 人工修 | 定義在 **symmetric** 空間；標記協定與 aseg 不同 |
| Hammers / LPBA40 / AAL | 30–40 位手工描繪配準成機率圖 | 標記協定不同 → 差異會混進 Dice |
| G-Node 公開資料 | 別人對同一模板跑過 recon-all | **未標示 FreeSurfer 版本** |

👉 **結論：自己跑，跟 167 顆同一套方法、同一個版本（7.4.1）。**
這也跟論文一致 —— TMI 2019 §V-A-1 明寫受試者與 atlas 都是 FreeSurfer 分割。

### 12.1 🔴 先補零，否則 recon-all 會內插

`recon-all` 會把輸入 conform 成 **256³ / 1mm / LIA**，並**保留影像中心**。

MNI152 三個邊長都是**奇數**，中心落在半整數體素上，跟目標的 128.0 差半格 —— **會觸發內插**：

```
軸 0　193 → 中心 96.5 　目標 128.0　偏移 −31.5　半整數 ✗
軸 1　229 → 中心 114.5　目標 128.0　偏移 −13.5　半整數 ✗
軸 2　193 → 中心 96.5 　目標 128.0　偏移 −31.5　半整數 ✗
```

`make_padded_atlas_input.py` 先補零成 256³（前 31/13/31、後 32/14/32），
補完三軸中心都是 **128.0，偏移為零**。

> FreeSurfer 的 log 顯示它**仍然跑了三線性內插** —— 但在整數座標上，三線性回傳的就是原值。

### 12.2 驗證：`verify_conform_roundtrip.py`

| 量測 | 值 | 若補零沒生效會是 |
|---|---|---|
| 共用遮罩的強度加權質心位移 | **0.0032 voxel** | 0.5 voxel |
| 相位相關次像素估計 | 0.0001（整數峰值在 0,0,0）| — |
| 轉回後與原圖的相關係數 | 0.999977 | — |

⚠️ **這支腳本一開始寫錯過**：原本用「各自算閾值遮罩」，兩張圖的遮罩因為 uchar 量化差了 1.4%，
量出 0.062 voxel 的**假 FAIL**。改成**共用同一個參考遮罩 + 強度加權質心**才對。

### 12.3 切回訓練空間：`make_atlas_seg.py`

aseg 回來之後是 256³，用**兩段整數切片**切回：

```
256³ ──[31:224, 13:242, 31:224]──> (193,229,193)   # 還原補零
     ──[0:192,   2:226,  0:192 ]──> (192,224,192)  # 跟 atlas 同一套裁切
```

**兩段都是整數切片，voxel 值一個都沒被動過。** 輸出 `IXI/atlas_mni152_09c_v3_seg.npz`。

驗證 5/5 通過，其中最重要的是：**評估用的 30 個結構全部都在，缺 0 個。**

📌 **品質對照**：atlas 的表面破洞 14/12，我們 166 顆受試者的中位數是 12/10 —— 幾乎就在中位數上。
原本擔心平均模板邊界模糊會讓拓撲重建失敗，結果沒有。耗時 2 小時 21 分。

---

## 13. Dice 評估與視覺化

### 13.1 `ASD/test_dice.py`

**`--test-dir` 必填，直接給路徑**（2026-09-13 改，見 §14 開頭）；atlas、atlas-seg、labels 有預設值。
tigerbx 組要另外給 `--atlas-seg IXI\atlas_mni152_09c_v3_seg_tigerbx.npz`。

三種模式：

```powershell
# ① 基準線：只有 Affine，模型無貢獻
python ASD\test_dice.py --baseline --test-dir data\mixed_preprocessed_v1\test --exp-name mix_exp1

# ② 單一模型
python ASD\test_dice.py --model models\mix_exp1\0230.pt --test-dir data\mixed_preprocessed_v1\test

# ③ 掃整個資料夾，畫 Dice 曲線
python ASD\test_dice.py --model-dir models\mix_exp1 --step 10 --test-dir data\mixed_preprocessed_v1\test
```

輸出 `dice_curve.csv`：`epoch, dice_mean, jneg_pct`。

🟢 **內建的 sanity check**：`epoch 0` 的 Dice 應該等於 `--baseline` 的值。
這兩個數字走**完全不同的程式路徑**，一致才代表評估鏈沒錯（實測都是 0.676014）。

⚠️ `--model` 要給 **`.pt` 檔**不是資料夾。給錯會噴 `PermissionError`（Windows 上開資料夾當檔案），
訊息看不出原因 —— 腳本已加參數檢查會先擋下來。

### 13.2 `ASD/visualize_dice.py`

```powershell
python ASD\visualize_dice.py --model models\mix_exp1\0230.pt --test-dir data\mixed_preprocessed_v1\test `
    --subject T053 --out-dir models\mix_exp1\vis_T053
```

- `--out-dir` 不給就存到模型旁邊的 `dice_vis\`（所有受試者擠在一起，建議分開給）。
- `--labels` 指定「評估哪些結構」，預設 repo 的 `labels.npz`（30 個 FreeSurfer 標籤），**平常不用動**。
  它決定 Dice 平均哪些結構、重疊圖畫哪些區域、長條圖列哪些結構；輪廓圖的 9 個結構是寫死的，跟它無關。
- tigerbx 組要換 `--test-dir data\tigerbx_preprocessed_v1\test` 和 `--atlas-seg IXI\atlas_mni152_09c_v3_seg_tigerbx.npz`，
  否則會拿 tigerbx 標籤去比 FreeSurfer 的 atlas 標籤，Dice 安靜地低掉約 0.14。

出三張：**標籤重疊**（紅=只有 atlas、綠=只有受試者、黃=重疊）、
**9 個結構的輪廓對照**、**逐結構長條圖**。上下排是「只有 Affine」vs「加上 VoxelMorph」。

另外 `draw-img/visualize_reg_ixi.py` 也能吃 ASD（它只讀 npz 的 `vol`），
出三平面 / 棋盤格 / 形變網格 / 疊圖 / Jacobian，但**要手動指定 `--atlas` 和 `--test-dir`**：

⚠️ 這支的 `--subject` **原本只吃完整 npz 路徑**，給純 ID 會噴
`ValueError: unknown filetype for A061` —— 訊息完全看不出原因。
已改成兩種都吃（ID 會到 `--test-dir` 底下找同名 npz），找不到時給可行動的訊息。

```powershell
python draw-img\visualize_reg_ixi.py --model models\asd_exp1\0190.pt `
    --atlas IXI\atlas_mni152_09c_v3.npz --test-dir data\ASD_preprocessed_v1\test `
    --subject T023 --out-dir models\asd_exp1\vis_T023 --gpu 0
```

**確認有沒有對上，實際要看哪裡**：

| 圖 | 對上 | 沒對上 |
|---|---|---|
| 棋盤格 | 方格交界處腦輪廓**接得起來** | 交界處錯開、像斷掉 |
| 標籤重疊 | 幾乎全黃 | 邊緣一圈明顯紅綠 |
| 形變網格 | 平滑的彎曲 | 打結、交叉、擠成一團 |
| Jacobian | 沒有 ≤0 的點 | 出現黑點＝折疊 |

📌 **最有判別力的是標籤重疊那張。** 三平面和棋盤格看的是灰階影像，腦大致對上就會看起來還好，
但結構邊界差幾 mm 目視分不出來 —— 那正是 Dice 在扣分的地方。

⚠️ 兩個容易踩的：
- 矢狀面**刻意偏離中線 28 voxel** —— 中線切不到海馬迴和殼核
- 檔名是 `<type>_<subject>_<epoch>.png`，**同 epoch 換受試者會互相覆蓋**，比較多顆要分 `--out-dir`

### 13.3 ✅ 訓練用的是 train 149（已確認）

**asd_exp1 訓練時 train = 149，A016_1 在訓練前就已經移除**（使用者在 B 台確認）。

A016_1 是前處理跑完、QC 的時候才發現的：皮質面積僅中位數 54%、表面破洞 83/71，
所以從訓練集移出（`A016_2` 保留）。**方法學要寫 149 / 17，不是 150 / 17。**
`log/asd_exp1_script.txt` 標頭寫的「train 150」是 `run_train.py:176` 的**寫死字串**，
不是執行時數的，**不能拿來當證據**。

👉 `run_train.py` 已改成執行時用 `glob` 實數，以後的 script log 才可信。

---

## 14. 加入第二包資料 / 混合訓練

**2026-09-07 起，資料一律放頂層 `data/`，`ASD/` 只留程式與筆記。**
原因：`ASD/` 這個名字指的是「FreeSurfer 這條線」而不是資料集，
一旦第二包資料放進去，路徑就開始說謊，而它是交接文件到處引用的位置。

```
data/
  ASD_data/{norm,aseg}/            原始 FreeSurfer 產物
  ASD_preprocessed_v1/{train,test} 前處理後的 npz
  <名稱>_data/{norm,aseg}/         新資料集照這個命名
  <名稱>_preprocessed_v1/
  mixed_preprocessed_v1/           make_mixed_set.py 併出來的
```

🔄 **2026-09-13 起一律直接給路徑，`--dataset` 已拿掉**：`--dataset ASD` 會自己組出
`data/ASD_preprocessed_v1`，指令上看不出是哪一版，有了 v2 之後還會安靜地拿 v1 去跑。

| 腳本 | 路徑參數 |
|---|---|
| `run_preprocess.py` | `--src-dir data\DGM_data --out-dir data\DGM_preprocessed_v1`（兩個都必填）|
| `make_mixed_set.py` | `--sources <各包的前處理資料夾> --out <輸出資料夾>` |
| `check_dataset.py` | `--dirs <前處理資料夾…>` |
| `find_duplicate_scans.py` | `--prep-dir <前處理資料夾>` |
| `run_train.py` | `--train-dir <…\train> --exp-name <實驗名>`（兩個都必填）|
| `test_dice.py` / `visualize_dice.py` | `--test-dir <…\test>` |

### 14.1 加一包新資料的完整流程

```powershell
# ① 放資料
#    data\NEWDS_data\fs_for_vxm\norm\<ID>.nii.gz
#    data\NEWDS_data\fs_for_vxm\aseg\<ID>.nii.gz
#    data\NEWDS_data\fs_stats\subjects.txt      一行一個 ID（或用 --subject-list 指定）

# ② 先驗一顆
python ASD\preprocess_fs.py --img-dir data\NEWDS_data\norm --seg-dir data\NEWDS_data\aseg `
    --atlas IXI\atlas_mni152_09c_v3.nii.gz --out-dir ASD\fs_check --only <某個ID>

# ③ 批次前處理
python ASD\run_preprocess.py --src-dir data\NEWDS_data --out-dir data\NEWDS_preprocessed_v1

# ④ 併成混合訓練用的一份
python ASD\make_mixed_set.py --sources data\ASD_preprocessed_v1 data\NEWDS_preprocessed_v1 `
    --out data\mixed_preprocessed_v2 --dry-run
python ASD\make_mixed_set.py --sources data\ASD_preprocessed_v1 data\NEWDS_preprocessed_v1 `
    --out data\mixed_preprocessed_v2

# ⑤ 訓練
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp2
```

### 14.2 `make_mixed_set.py` 在做什麼

`train.py` 是 `glob(datadir/*.npz)`，**只吃單一扁平資料夾**，
所以混合訓練必須先把各資料集的 npz 併到同一個目錄。

- 🔄 **預設是實體複製**（2026-09-08 使用者決定改掉硬連結）：Windows 檔案總管看不出哪個檔案是硬連結，
  省 2.2 GB 不值得一個看不見的機關，尤其這份資料要在機器之間搬。真的要省空間再加 `--hardlink`。
- **train 併 train、test 併 test**，各來源自己的切分原封不動。
  所以只要每個資料集的切分是受試者層級的，合併後仍然沒有 leakage。
- 🔴 **會先檢查撞名**。不同資料集若有同名受試者（例如兩邊都有 `A001.npz`），
  直接併會**安靜地少一顆**。偵測到就中止，要加 `--prefix` 讓檔名帶資料集名稱。
- 產出 `mixed_manifest.json`，記錄每個檔案來自哪個資料集 —— 之後要拆開分析用得到。

### 14.3 🔴 混合訓練的前提

| 項目 | 要求 | 為什麼 |
|---|---|---|
| **atlas** | 必須同一顆 | 不同 atlas 的 npz 空間對不起來 |
| **前處理** | 必須完全一致 | 亮度正規化、去顱骨、Affine 設定不同會變成 confound |
| **影像來源** | 全體一致（都 `norm.mgz` 或都 `brain.mgz`）| 見 §4 |
| **標籤方法** | 必須同一套 | **見下** |

🔴 **標籤方法不一致本身就是 confound。**
若一批用 aseg、另一批用 SynthSeg，兩邊的 Dice **不能放同一張表比** ——
差異裡混雜了「配準品質」和「兩套分割方法」，分不開。

📌 **目前規劃的新資料同樣是 FreeSurfer 切出來的**（使用者 2026-09-07 確認），
所以這一條是滿足的。但**版本要對齊**：我們是 **7.4.1**。
不同版本的 `autorecon3` 會影響標籤 2 / 3 / 41 / 42（大腦白質與皮質），
而那四個都在評估用的 30 個結構裡。

### 14.4 混合之後要重新確認的事

- [ ] **基準線要重算。** `--baseline` 是在該批 test 上算的，換了 test 集就是不同的數字，
      不能沿用 ASD 的 0.676014。
- [ ] **切分仍然要是受試者層級。** 若兩包資料有同一個人（跨機構收案有可能），
      合併後會 leakage —— `make_mixed_set.py` 只擋得住**檔名相同**的情況，
      擋不住「同一人但兩邊 ID 不同」。
- [ ] **兩批的年齡 / 族群分布。** ASD 這批 eTIV 中位數 1.29 L，可能包含兒童（§10）。
      若新資料是成人，混合訓練會讓模型同時看到兩種尺度 —— 不一定是壞事，但要寫進方法學。

> 🔄 **2026-09-08 結果**：① 基準線重算為 **0.6753**（28 位 test）；② 同一人用 DICOM ＋ 標籤 Dice 查過，
> DGM 內有兩對，已綁在同一邊（§15）；③ 年齡：ASD 中位數 22 歲（5–34，未滿 18 歲 24 位）、
> DGM 23 歲（19–43）、VNT 36 歲（22–48）。ASD 確實有兒童，但主體是成人。

---

## 15. 資料把關：DICOM 檔頭與同一人（2026-09-07）

FreeSurfer 端逐一讀了 **4,903 個 DICOM 序列**的檔頭，推翻了幾件事，也找出新的問題。

### 15.1 ASD 最新的排除／納回表（170 → 164）

| 受試者 | 問題 | 處置 |
|---|---|---|
| A043 | 雜訊過高、灰白對比不足 | 排除 |
| T085 | 只有 120 / 192 張切片，來源即缺 | 排除 |
| A016_1 | 皮質面積僅中位數 54%；技師當場標註 low contrast | 排除 |
| A016_2 | ID 是 QA＋日期、性別欄 O：品管掃描，不是受試者 | 排除 |
| YT13 | 與 A0131 的掃描時間**精確到秒相同**：同一次掃描被匯出成兩個 ID | 排除 |
| A0132 | 與 A0131 是同一位 5 歲男童，相隔 23 天的第二次掃描 | 排除 |
| T065 | 建檔時誤植為 T056；資料夾內留有說明檔 | **納回** |
| A012 | 資料夾混了兩次掃描 | 已分離重跑，納入 |

- A013 是另一個人（23 歲成人），跟 A0131 無關。
- **現行清單是 `data/ASD_data/fs_stats/subjects.txt`（164 個，隨資料附）**。`ASD/subjects_final.txt` 是 08-23 的舊版，不要再用。
- 切分：ASD train 148 / test 16。
- 🔒 個資（出生日期／身高／體重／精確掃描日）**不進公開版控**，只記判定依據的種類。

**對舊結果的影響**：asd_exp1 的 17 顆 test 裡，A0131 反而是最低的（0.740）。
排除它之後平均只升 +0.0026，小於標準誤 0.0038 → **這次洩漏沒有把分數灌高**，但方法學已經修正。

### 15.2 用影像找同一人：`find_duplicate_scans.py`

DICOM 的人口學欄位在這批**不可全信**：技師會複製上一位的登錄資料（D023/D024、T029/T028 都是實例）。
所以另外用影像檢查：兩顆腦 affine 到 atlas 之後，量 30 個結構的標籤 Dice。

| 配對 | 標籤 Dice | 判定 |
|---|---|---|
| A0131 / YT13 | 0.9793 | 同一次掃描（重複匯出）|
| D015 / D037 | 0.8561 | 同一人，相隔 3 個月 |
| D038 / DGM002 | 0.8528 | 同一人，相隔 9 個月 |
| A0131 / A0132 | 0.7318 | 同一人（5 歲），相隔 23 天 |
| D018 / DGM001 | 0.6995 | 不同人（DICOM 定案）|
| VNT027 / VNT028 | 0.6609 | 不同人（複製登錄）|

不同人的分布：ASD 13,366 對（中位數 0.663、第 99 百分位 0.721、最大 0.747）、
DGM 1,431 對（0.656 / 0.722）、VNT 2,278 對（0.653 / 0.719）。

- **抓得到**：同一次掃描（0.98）、成人同一人（0.85）
- **抓不到**：5 歲兒童相隔 23 天只有 0.73，落在不同人的分布裡面
- 🔴 **教訓**：一開始拿 1 對同人 vs 8 對不同人定門檻 0.70，全掃之後誤報幾百對。小樣本定的門檻不能用。
- 判不出來時採保守做法：**當成同一人放在一起**（錯了只損失一點切分自由度，反過來錯就是 leakage）。

### 15.3 DGM 與 VNT

- **DGM**：D015 / D037、D038 / DGM002 是同一人 → 54 個掃描 = **52 人**，寫成 `ASD/DGM_groups.txt` 給 `--group-map`。
  D015 / D037 的性別欄相反，至少一邊是錯的，**這兩顆的性別欄不要用於分析**。
- **VNT**：68 顆確認是 68 個獨立個體。
- `preprocess_fs.py` 的檔名歸戶規則已收緊：原本 VNT001～VNT009 會被誤判成同一人（字根 VNT00）。

---

## 16. 三包混合訓練：mix_exp1（2026-09-08）

### 16.1 資料

| | ASD | DGM | VNT | 合計 |
|---|---|---|---|---|
| 受試者 | 164 | 54 掃描 = 52 人 | 68 | 286 掃描 |
| 年齡中位數（範圍）| 22（5–34）| 23（19–43）| 36（22–48）| |
| 未滿 18 歲 | 24 | 0 | 0 | |
| train / test | 148 / 16 | 49 / 5 | 61 / 7 | **258 / 28** |

**為什麼可以合併**：同一台 Skyra 3T、同一個 MPRAGE 協定；三包內部「不同人」之間的標籤 Dice 分布幾乎一樣
（中位數 0.663 / 0.656 / 0.653）。各包自己的切分原封不動併起來，所以仍然是受試者層級。

```powershell
# 當時用的是舊的 --dataset 寫法，以下換成 2026-09-13 起「直接給路徑」的寫法
python ASD\run_preprocess.py --src-dir data\DGM_data --out-dir data\DGM_preprocessed_v1   # 自動套用 ASD\DGM_groups.txt
python ASD\run_preprocess.py --src-dir data\VNT_data --out-dir data\VNT_preprocessed_v1
python ASD\make_mixed_set.py --sources data\ASD_preprocessed_v1 data\DGM_preprocessed_v1 data\VNT_preprocessed_v1 `
    --out data\mixed_preprocessed_v1
python ASD\check_dataset.py --check --dirs data\ASD_preprocessed_v1 data\DGM_preprocessed_v1 `
    data\VNT_preprocessed_v1 data\mixed_preprocessed_v1          # 搬到 B 台之後
python ASD\run_train.py --train-dir data\mixed_preprocessed_v1\train --exp-name mix_exp1
python ASD\test_dice.py --baseline --test-dir data\mixed_preprocessed_v1\test --exp-name mix_exp1
python ASD\test_dice.py --model-dir models\mix_exp1 --step 10 --test-dir data\mixed_preprocessed_v1\test
```

⚠️ 搬到 B 台時曾經**複製不完整**（191 / 258 個 npz），`check_dataset.py` 補上混合集對帳後才抓到。

### 16.2 結果

| 項目 | 值 |
|---|---|
| 基準線（只做線性對位）| **0.6753**（epoch 0 的值與 `--baseline` 完全一致 → 評估流程沒錯）|
| 最佳 epoch 230 | **0.7874**（+0.1121），折疊率 0.000% |
| epoch ≥ 90 | 平均 0.7825，全距 0.0106 → 各檢查點分不出高下，「最佳 epoch」沒有特別意義 |
| 受試者之間的標準差 | 0.040 → 0.015：模型把大家拉到 0.79 附近 |
| 改善幅度 vs 起點 | 相關係數 **−0.96**：起點越低進步越多 → **該報模型後的絕對值，不是改善幅度** |

分資料集（模型後）：ASD 0.787、DGM 0.790、VNT 0.786，三包幾乎一樣；VNT 起點最低（0.650，年紀最大），所以進步最多。
DGM 5 位、VNT 7 位，人數太少，只作觀察。

跟 asd_exp1 相比：資料多 73%（149 → 258），但 test 集不同，Dice 不能直接比；進步幅度小於標準誤。

---

## 17. tigerbx 標籤對照：tiger_exp1（2026-09-09）

同一批 286 位、同一個切分，只把 **FreeSurfer 的影像與標籤換成 tigerbx 的**。

### 17.1 怎麼比才公平

1. **同一批人、同一個切分**：`--split-from data\mixed_preprocessed_v1\mixed_manifest.json`，286 位逐一比對歸屬相同
2. **補做偏場校正**：tigerbx 的 `_tbet` 是原始強度（白質變異係數 15.9%），FreeSurfer 已做過 nu 校正 → 加 `--n4`
3. **atlas 的標籤也換成 tigerbx**：兩套方法在 atlas 的 30 個結構上只有 0.855 一致。沿用 FreeSurfer 的 atlas 標籤，
   光 atlas 這一端就先扣掉約 0.14，那跟配準無關
4. **同一人綁同一邊**：DGM 那兩對在兩組都放同一邊

仍然存在的差異：去顱骨鬆緊不同（腦遮罩是有標籤腦區的 FreeSurfer 1.53 倍、tigerbx 1.23 倍）；FreeSurfer 組多一次重採樣。
👉 **比的是兩條完整流程，不是兩個分割演算法。**

```powershell
# atlas 的 tigerbx 分割（data\tigerbx\atlas\ 是 tigerbx 對補零 256³ atlas 的輸出）
python ASD\make_atlas_seg.py --src data\tigerbx\atlas\mni152_09c_t1_padded256_aseg.nii.gz `
    --out IXI\atlas_mni152_09c_v3_seg_tigerbx.npz

# 前處理（依 log\tigerbx_preprocess.txt 還原；當時直接呼叫 preprocess_fs.py，沒有 script log）
python ASD\preprocess_fs.py --img-dir data\tigerbx_data\fs_for_vxm\norm --seg-dir data\tigerbx_data\fs_for_vxm\aseg `
    --atlas IXI\atlas_mni152_09c_v3.nii.gz --out-dir data\tigerbx_preprocessed_v1 `
    --subject-list data\tigerbx_data\fs_stats\subjects.txt --list-is-final --grouping none `
    --group-map ASD\DGM_groups.txt --split-from data\mixed_preprocessed_v1\mixed_manifest.json --n4

# 評估一定要換 atlas 分割
python ASD\test_dice.py --test-dir data\tigerbx_preprocessed_v1\test `
    --atlas-seg IXI\atlas_mni152_09c_v3_seg_tigerbx.npz --model-dir models\tiger_exp1 --step 10
```

### 17.2 結果

| | 基準線 | 模型後 | 模型貢獻 | 折疊率 |
|---|---|---|---|---|
| FreeSurfer 組（mix_exp1，ep 230）| 0.6753 | 0.7874 | +0.1121 | 0.000% |
| tigerbx 組（tiger_exp1，ep 240）| 0.7376 | 0.8594 | +0.1218 | 0.000% |

- 逐人配對：tigerbx 組的貢獻平均多 **+0.0097**（配對標準誤 0.0026，約 3.7 倍），28 位中 22 位較大 → 測得到，但幅度很小
- 絕對值差 0.072，其中 0.062 在訓練前就存在（tigerbx 標籤較平滑）→ **絕對值不能直接比**
- 30 個結構全部是 tigerbx 組較高：差最多的是脈絡叢（+0.25，FreeSurfer 本身就切不穩）、殼核（+0.12）、蒼白球（+0.11）；
  差最少的是大腦白質（+0.015）。反映的是標籤性質，不代表配準變好
- tiger_exp1 曲線上最高其實是 epoch 230（0.8598），留存與評估用的是 0240（0.8594），差距在雜訊內

### 17.3 A0131（5 歲，兩組都是最低）

- 排除三個假設：體積切錯（兩方法體積比 0.98）、結構太小、影像品質
- **右側是標籤不確定**：兩套分割的右杏仁核一致性 0.525（286 位中最低）；換成 tigerbx 標籤，Dice 從 0.476 回到 0.870
- **左側比較像配準困難**：兩方法有共識（0.751），模型卻只有 0.394（FreeSurfer 組）
- **不是「兒童較差」**：24 位未成年的分割一致性與成人相當（0.83 vs 0.84）
- 換成 tigerbx 標籤後，跟第二低的差距從 0.04 縮到 0.002

---

## 18. 跟論文比（Balakrishnan et al., IEEE TMI 2019）

### 18.1 論文 Table I 全表，接上我們

| 方法 | Dice（標準差）| GPU 秒 | CPU 秒 | 折疊 voxel 數 | 折疊率 % |
|---|---|---|---|---|---|
| *作者：8 個公開資料集 3,731 顆，test 250 · TitanX / Xeon E5-2680* | | | | | |
| Affine only | 0.584 (0.157) | 0 | 0 | 0 | 0 |
| ANTs SyN (CC) | 0.749 (0.136) | – | 9059 (2023) | 9662 (6258) | 0.185 (0.091) |
| NiftyReg (CC) | 0.755 (0.143) | – | 2347 (202) | 41251 (14336) | 0.793 (0.208) |
| **VoxelMorph (CC)** | **0.753 (0.145)** | 0.45 (0.01) | 57 (1) | 19077 (5928) | 0.366 (0.114) |
| VoxelMorph (MSE) | 0.752 (0.140) | 0.45 (0.01) | 57 (1) | 9606 (4516) | 0.184 (0.087) |
| *我們：286 位，test 28 · RTX 4060 Laptop / 單線程 CPU* | | | | | |
| 只做線性對位 · FreeSurfer | 0.675 (0.152) | 0 | 0 | 0 | 0 |
| VoxelMorph (CC) · FreeSurfer | 0.787 (0.122) | 0.31 (0.00) | 11.3 (0.1) | 0 | 0 |
| 只做線性對位 · tigerbx | 0.738 (0.137) | 0 | 0 | 0 | 0 |
| VoxelMorph (CC) · tigerbx | 0.859 (0.076) | 0.31 (0.00) | 11.3 (0.1) | 0 | 0 |

- **標準差用論文的算法**：「across structures and subjects」＝ 28 位 × 30 個結構的 840 個 Dice 攤平後算
- **時間**：論文 §V-B「1) Runtime」寫明是 Xeon E5-2680 ＋ TitanX，計時從線性對位之後開始。
  我們量的是 `model(scan, atlas, registration=True)`（含積分），GPU 暖機 2 次後量 28 顆、CPU 單線程量 3 顆。
  論文**沒寫 CPU 線程數**；硬體不同，影像也比論文大 20%（192×224×192 vs 160×192×224）→ 只能看數量級
- **折疊**：論文分母是腦內 520 萬 voxel；我們兩組 28 位都是 0 個，主要因為用了微分同胚版

### 18.2 論文其他跟我們有關的數字

**Table II（專家手動標註，Buckner40 共 39 顆；模型沿用 Table I 那批，沒有重訓）**

| 方法 | Dice（標準差）| 比 Affine 多 |
|---|---|---|
| Affine only | 0.608 (0.175) | — |
| ANTs SyN (CC) | 0.776 (0.130) | +0.168 |
| NiftyReg (CC) | 0.776 (0.132) | +0.168 |
| VoxelMorph (MSE) | 0.766 (0.133) | +0.158 |
| VoxelMorph (MSE) inst. | 0.776 (0.132) | +0.168 |
| VoxelMorph (CC) | 0.774 (0.133) | +0.166 |
| VoxelMorph (CC) inst. | 0.786 (0.132) | +0.178 |

inst. ＝ 每一對再用梯度下降微調 100 次（GPU 23.7 秒、單線程 CPU 628 秒）。
作者換標籤後絕對值 0.753 → 0.774、貢獻 +0.169 → +0.166 幾乎不變；跟我們換 tigerbx 的觀察方向一致（但作者連受試者也換了）。

**Fig. 7（λ 敏感度，validation Dice）**：論文只有圖，下面是從圖上量的，誤差約 ±0.001。

| λ（CC）| 0 | 0.5 | **1** | 1.5 | 2 | 5 |
|---|---|---|---|---|---|---|
| Dice ≈ | 0.687 | 0.743 | **0.746** | 0.746 | 0.745 | 0.737 |

| λ（MSE）| 0 | 0.005 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|---|
| Dice ≈ | 0.688 | 0.742 | 0.744 | 0.745 | 0.734 |

作者的做法：**試多個 λ，每個 λ 各訓練一個模型**，挑 validation Dice 最高的去跑 test。
論文沒寫 Table I 最後用哪個 λ；CC 在 λ = 1 和 1.5 並列最高。我們只訓練一個模型、直接用 λ = 1.0。
Table III（受試者對受試者）、Table IV（訓練時加入分割標籤）的設定跟我們不同，沒有拿來比。

### 18.3 逐項差異

| 項目 | 作者 | 我們 |
|---|---|---|
| 資料 | 8 個公開資料集、3,731 顆；多中心，年齡、疾病、掃描參數各異 | 3 包 286 位；同一台 Skyra 3T、同一協定 |
| 切分 | train 3,231 / val 250 / test 250 | train 258 / val 0 / test 28 |
| 同一人 | 未說明 | DICOM ＋ 標籤 Dice 檢查，同一人放同一邊 |
| 去顱骨 | FreeSurfer | FreeSurfer（norm.mgz）；tigerbx 組用 tigerbx |
| 線性對位 | FreeSurfer 做；**對到哪個模板論文沒寫** | ANTs Affine，直接對到評估用的 MNI152 atlas |
| atlas | 外部資料集的平均模板，檔案與參數沒公布 | MNI152 2009c；標籤自己跑 FreeSurfer／tigerbx |
| 影像大小 | 160 × 192 × 224 | 192 × 224 × 192 |
| **形變場** | **位移場（Table I 主結果，= int_steps 0）** | **速度場積分，微分同胚（int_steps 7）** |
| 形變場解析度 | 全解析度 | 半解析度（int_downsize 2）|
| 網路／損失 | U-Net（Fig. 3）；CC 窗格 9 ＋ λ × 梯度平滑 | 相同（repo 預設）|
| λ | 用 validation 挑 | 1.0，沒自己挑 |
| 訓練量 | 預設 150,000 次迭代 | 250 epoch × 100 步 = 25,000 次（約 1/6），epoch 90 後已持平 |
| 優化器 | ADAM，lr 1e-4，batch 1 | 相同 |
| 選模型 | validation Dice | 看 test 曲線挑 epoch（偏樂觀）|
| 框架 | Keras + TensorFlow | PyTorch |

### 18.4 報告時的說法

- **絕對值我們比較高，但起點就高了約 0.09**（0.675 vs 0.584）。看模型貢獻，我們 +0.112、作者 +0.169，我們比較低
- 起點越高，剩下能進步的空間越小，但這個天花板我們量化不出來
- **折疊率不能比**：形變場版本不同（見 §9.1 更正）
- 資料、atlas、線性對位三個起點都不同 → 只能比「模型貢獻」這類相對量，而且也只能參考

📌 29 頁簡報的產生器在 `ASD/slides_src/2026-09_mix_tigerbx/`（用法見該資料夾的 README.md）。
投影片上的數字都由 `gather.py` 從 `models/*/dice_*.csv` 算出，不手打。
