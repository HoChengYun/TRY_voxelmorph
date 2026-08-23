# ASD 資料集：FreeSurfer 產物 → VoxelMorph 訓練實作手冊

何承運 · 2026
建立日期：**2026-08-23**

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
> ⚠️ **目前狀態：程式已就緒並驗證通過，卡在資料尚未送到 Windows。**

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
| `recon-all` 跑完 | **167**（9 天 17 小時） |
| 暫定可用 | **166**（167 − T065 身分待確認） |

⚠️ **這批不是 IXI。** IXI 是健康成人（20–86 歲、三台機器），ASD 這批是另一個 cohort。
兩者混用或跨批比較的取捨見 §8。

---

## 2. 🔴 資料品質：排除清單

來源：`D:\MyHome\MRI\FreeSurfer\docs\ASD_資料品質記錄.md`（FreeSurfer 端已查證，有 log／檔案證據）

| 受試者 | 問題 | 處置 |
|---|---|---|
| **A043** | 影像雜訊過高、灰白對比不足（白質只認出約 5%）| 🔴 排除（recon-all 未完成）|
| **T085** | 只有 120/192 張切片（缺 0001–0072）+ 首張檔案截斷 | 🔴 排除（來源即缺）|
| **A012** | 資料夾混了兩次掃描（0801 + A012 各 192 張 → 合併成 384）| 🟡 修好前不要用 |
| **T065** | 資料夾名 T065，但 DICOM 病人 ID 是 **T056** | 🟠 身分待確認，先不納入 |

這四個已寫死在 `preprocess_fs.py` 的 `DEFAULT_EXCLUDE`，當作保險（暫定清單本來就已扣除）。

### ⚠️⚠️ 尚未完成的檢查：混掃描全面掃描

**A012 是因為疊成 384 層超過 FreeSurfer 的 256 上限才報錯露餡。**
如果某個資料夾也混了兩次掃描、但總層數剛好塞得進 256，**`recon-all` 不會報錯**，
會安靜跑出一顆「兩個人疊在一起」的腦。這種資料餵進訓練會讓模型學到不存在的解剖結構。

**狀態：尚未開始。** 那批 .IMA 在 Ubuntu VM 上，FreeSurfer 端的 session 沒有 VM 存取權，
指令必須由使用者在 VM 執行。判讀標準（FreeSurfer 端提供）：
> 某資料夾內「不同 DICOM ID > 1」或「張數 > 200」→ 混了不只一次掃描，
> 該受試者已跑完的結果不可信，要清乾淨重跑。

🔒 **在這項完成之前，`preprocess_fs.py` 會拒絕批次執行**（見 §5 的閘門）。

---

## 3. 受試者歸戶（避免 data leakage）

同一個人的多次掃描若一個進 train、一個進 test，模型等於看過答案，Dice 會虛高。
**切分必須以「受試者」為單位，不是檔案層級 shuffle。**

### 已知的多次掃描

| 組別 | 檔名 | 狀態 |
|---|---|---|
| A016 | `A016_1`, `A016_2` | 底線後綴，**程式預設規則會自動合併** |
| A013 | `A013`, `A0131`, `A0132` | 🟠 **待老師確認**，程式**不會**自動合併 |

**為什麼 A013 那組不自動合併**：`A0131` 可能是「A013 的第 1 次掃描」，也可能就是編號 A0131
的獨立受試者——而且 **`A013` 本身也存在**。光靠字串規則無法判斷，猜錯會安靜地造成錯誤切分。

✅ **已用程式掃過全部 166 個 ID**（偵測「只差結尾一個數字」與「某 ID 是另一 ID 的前綴」兩種型態），
結論：**這種曖昧全批只有 `A013` 這一組，沒有第二處。**
→ 要問老師的只有這一件事，不需要逐一比對 166 個。

### 確認後怎麼填

寫一個 `ASD/groups.txt`（TSV，一行一組），程式用 `--group-map` 讀：

```
A013	A013
A0131	A013
A0132	A013
```

沒列到的 ID 走預設規則（去掉結尾的 `_<數字>`）。

---

## 4. 從 FreeSurfer 端轉檔過來

### 4.1 在 Ubuntu VM 上執行（FreeSurfer 環境）

```bash
export SUBJECTS_DIR=/home/cheng/workspace/subjects/ASD_result
OUT=/mnt/hgfs/outside/fs_for_vxm        # ← 寫到共享資料夾，Windows 才看得到
mkdir -p $OUT/brain $OUT/aseg

for d in $SUBJECTS_DIR/*/; do
  s=$(basename "$d")
  grep -q "finished without error" "$d/scripts/recon-all.log" 2>/dev/null || continue
  mri_convert "$d/mri/brain.mgz" "$OUT/brain/${s}.nii.gz"
  mri_convert "$d/mri/aseg.mgz"  "$OUT/aseg/${s}.nii.gz"  -rt nearest
done
```

- 只轉「跑成功」的受試者
- **影像一律用 `brain.mgz`**（已 N4 + 去顱骨 + 亮度正規化），不要混 `norm.mgz`
- 容量估計：每人約 5–7 MB，166 人約 **1–1.5 GB**

### 4.2 搬到 Windows

`/mnt/hgfs` 是 VMware 共享資料夾的掛載點，所以寫進 `/mnt/hgfs/outside/` 後 Windows 端就看得到。

> ⚠️ **待填**：`/mnt/hgfs/outside` 對應的 Windows 路徑 = `______________`

⚠️ **VMware 共享資料夾對大量檔案複製不太可靠**（T085 最初就被懷疑是複製出問題，
後來確認是來源本身缺檔）。搬完務必核對檔案數與大小。
`preprocess_fs.py` 會列出「白名單裡有但 `--brain-dir` 找不到」以及「缺對應 aseg」的 ID，
可以當作核對工具。

### 4.3 放到哪

```
ASD/
├── brain/          ← brain.mgz 轉出的 .nii.gz
├── aseg/           ← aseg.mgz 轉出的 .nii.gz
├── subjects_final.txt   ← FreeSurfer 端給的最終清單
└── groups.txt           ← 歸戶對照（老師確認後）
```

> 📌 `.gitignore` 已設定：`ASD/` 底下的**資料檔會被忽略**，但 `*.py` / `*.md` /
> **根目錄的 `*.txt`**（清單、歸戶表）會進版控。清單是重要溯源資料，不要放進子資料夾。

---

## 5. 前處理：`preprocess_fs.py`

### 5.1 和 IXI 的差別

| 步驟 | IXI (`preprocess_ixi.py`) | ASD (`preprocess_fs.py`) |
|---|---|---|
| N4 bias correction | ✅ 做 | ❌ **跳過**（FreeSurfer 的 `nu.mgz` 階段已做）|
| 去顱骨 | ✅ antspynet（最慢的一步）| ❌ **跳過**（`brain.mgz` 已去過）|
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
只允許 `--dry-run` 和 `--only <SUBJECT>`。原因就是 §2 的混掃描檢查尚未完成。

FreeSurfer 端已承諾：**在該檢查完成前，不會交付任何標成 final 的清單。**

### 5.3 執行順序

**① 先看歸戶與切分（不動影像）**
```powershell
python ASD\preprocess_fs.py `
    --brain-dir ASD\brain --seg-dir ASD\aseg `
    --atlas IXI\atlas_mni152_09c_v3.nii.gz `
    --out-dir ASD\ASD_preprocessed_v1 `
    --subject-list ASD\subjects_final.txt `
    --dry-run
```

**② 驗證 1 顆**（產生 nii 供 ITK-SNAP / Freeview 目視確認）
```powershell
python ASD\preprocess_fs.py `
    --brain-dir ASD\brain --seg-dir ASD\aseg `
    --atlas IXI\atlas_mni152_09c_v3.nii.gz `
    --out-dir ASD\fs_check --only A001 --save-nii
```
目視要確認三件事：
1. 影像有對到 atlas 空間、shape = (192, 224, 192)
2. **標籤和影像完全疊合**（最重要）
3. 標籤值仍是整數，沒有出現奇怪的中間值

**③ 拿到最終清單後批次跑**
```powershell
python ASD\preprocess_fs.py `
    --brain-dir ASD\brain --seg-dir ASD\aseg `
    --atlas IXI\atlas_mni152_09c_v3.nii.gz `
    --out-dir ASD\ASD_preprocessed_v1 `
    --subject-list ASD\subjects_final.txt --list-is-final `
    --group-map ASD\groups.txt `
    --save-nii > .\log\asd_preprocess.txt 2>&1
```

### 5.4 輸出

```
ASD/ASD_preprocessed_v1/
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

## 7. 訓練設定

atlas、shape、正規化方式全部沿用 IXI 那條線，所以指令跟 `CLAUDE.md` 的一樣，
只是 `datadir` 換成 ASD 的。

```powershell
python voxelmorph-code\scripts\torch\train.py ASD\ASD_preprocessed_v1\train `
    --atlas IXI\atlas_mni152_09c_v3.npz `
    --model-dir models\asd_expN `
    --epochs 250 --gpu 0 `
    --image-loss ncc --lambda 1.0 > .\log\asd_expN.txt 2>&1
```

🔴 **`--image-loss` 預設是 `mse`**，想跑 NCC 一定要明寫。

🔴 **λ 的尺度取決於 `--image-loss`**：論文對 CC 的最佳值是 **≈1–2**，對 MSE 是 0.01–0.02。
`train.py` 預設的 0.01 是為 MSE 設的，**搭 NCC 用等於幾乎沒有正則化**。
IXI 那邊的 exp3–exp8 就是踩了這個（平滑項只佔損失 1–3%）。詳見 `CLAUDE.md`「超參數」節。

📌 **log 檔名用 `asd_expN` 前綴**，跟 IXI 那邊的 `expN` 區隔。
**每跑一個實驗，指令一定要存成 `log/asd_expN_script.txt`**（UTF-8）。

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
| 第二段 | ASD（166）| aseg | **external validation**，明講跨 cohort / 跨機器 |

好處：保住既有的 581 顆 IXI 和訓練好的模型；ASD 樣本數少，拿去當獨立驗證剛好。
而且如果表現掉了，那本身就是**泛化能力的證據**，是有價值的結論而不是瑕疵。

### 另一個選項

ASD 這批自己 train + test，Dice 內部一致。缺點是樣本數少（166 → train 150 / test 16），
且犧牲了既有的 IXI 主結果。

---

## 9. 實驗記錄

> 尚未開始。每跑一個補一列，並存 `log/asd_expN_script.txt`。

| 實驗 | 資料 | atlas | epochs | image-loss | λ | int-steps | int-downsize | 最佳 epoch | Dice | %\|J\|≤0 | 備註 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| （待填）| | | | | | | | | | | |

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

### 🔴 擋住開跑的

- [ ] **混掃描全面檢查**（§2）—— 需在 Ubuntu VM 執行，完成後取得**最終清單**
- [ ] **`/mnt/hgfs/outside` 對應的 Windows 路徑**（§4.2）
- [ ] **資料實際搬到 `ASD/brain/`、`ASD/aseg/`**，並核對檔案數與大小

### 🟠 需要問老師

- [ ] **A013 / A0131 / A0132 是否同一人**（§3）—— 全批唯一曖昧處
- [ ] **T065 的真實身分**（資料夾名 T065 但 DICOM ID 是 T056）

### 🟡 研究設計

- [ ] **Dice 報在哪批**（§8）—— 兩段式 vs ASD 自己 train/test

### 程式面

- [ ] **Dice 評估腳本尚未撰寫**（`test_ixi.py` 只算 NCC / SSIM）
- [ ] 資料到位後跑 §5.3 的三步驟

---

## 11. 相關檔案

| 檔案 | 用途 |
|---|---|
| `ASD/preprocess_fs.py` | FreeSurfer 產物 → npz（含 seg）|
| `ASD/verify_seg_transform.py` | 驗證 affine 共用 + 最近鄰內插 |
| `ASD/subjects_final.txt` | 最終可用清單（待 FreeSurfer 端交付）|
| `ASD/groups.txt` | 受試者歸戶對照（待老師確認）|
| `CLAUDE.md` | 專案總覽、超參數、實驗記錄 |
| `FreeSurfer_到_VoxelMorph_交接.md` | FreeSurfer 端寫的接入說明 |
| `D:\MyHome\MRI\FreeSurfer\docs\ASD_資料品質記錄.md` | 品質問題的原始診斷 |
| `D:\MyHome\MRI\FreeSurfer\docs\ASD_可用清單_暫定.txt` | 暫定清單（166，**非最終版**）|
| `D:\MyHome\MRI\FreeSurfer\docs\ASD_全部資料夾清單.txt` | 全部 170 個，含壞掉的 |
