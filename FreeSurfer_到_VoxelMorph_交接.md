# 交接：把 FreeSurfer 的產物接進你們的 VoxelMorph 訓練

> **寫給**：在 `C:\Users\h4524\claude_cheng` 做 VoxelMorph 的人／AI
> **寫的人**：另一個 Claude session（負責 FreeSurfer 那條線，工作目錄 `D:\MyHome\MRI\FreeSurfer`）
> **日期**：2026-08-23
>
> 我讀過你們的 `IXI/preprocess_ixi.py`、`IXI_preprocessed_v3/` 的實際 npz 內容，
> 以下都是對照你們**現有格式**寫的，不是通用建議。

---

## 0. 一句話結論

> **你們 pipeline 的前 3 步（N4 → 去顱骨 → 正規化），FreeSurfer 全都做過了。**
> 接進來只要：**轉檔 → 沿用你們的 Affine 對位 → 打包**。
> 而且 FreeSurfer 多送你們一樣現在沒有的東西：**分割標籤（可以算 Dice）**。

---

## 1. 兩邊現況

### 你們現在的 pipeline（我讀 `preprocess_ixi.py` 得到的）

```
原始 IXI .nii.gz
  ↓ ① ants.n4_bias_field_correction()
  ↓ ② antspynet.brain_extraction()  去顱骨
  ↓ ③ ants.registration(type_of_transform='Affine')  對位到 atlas_mni152_09c_v3
  ↓ ④ percentile(1,99) clip → min-max 正規化到 [0,1]
  ↓ ⑤ np.savez_compressed(dst, vol=img_np)
train/ (522) + test/ (59)
```

**npz 現況**（我實際載入 `IXI002-Guys-0828-T1.npz` 確認）：
```
keys : ['vol']
vol  : shape=(192, 224, 192)  dtype=float32  範圍 [0.0, 1.0]
```

### FreeSurfer 這邊有什麼

已對 **167 位受試者**（ASD 研究資料，Siemens Skyra 3T，MPRAGE 1mm）跑完完整 `recon-all -all`。
每位在 `$SUBJECTS_DIR/<受試者>/mri/` 底下有：

| 檔案 | 內容 | 空間 |
|---|---|---|
| `orig.mgz` | conform 後的原始影像 | 256³ / 1mm / LIA |
| `nu.mgz` | **N4 偏場校正後** | 同上 |
| `T1.mgz` | 強度正規化後（白質≈110）| 同上 |
| `brainmask.mgz` | **去顱骨後** | 同上 |
| **`norm.mgz`** | 去顱骨 + atlas 正規化 ⭐ **實際採用這個** | 同上 |
| `brain.mgz` | 去顱骨 + aseg 正規化（多一道，未採用）| 同上 |
| **`aseg.mgz`** | **皮質下分割標籤（~40 結構）** ⭐ 你們缺的就是這個 | 同上 |
| `transforms/talairach.xfm` | 仿射矩陣（→ **MNI305**）| — |

> **conformed 空間**＝ FreeSurfer 把每個人統一成 **256×256×256、1mm 等向、LIA 方向**。
> ⚠️ 注意：conform **只統一了「箱子」，沒有對齊「腦」**——每個人的腦在箱子裡的位置／角度／大小仍不同。

---

## 2. ⭐ 對照表：你們哪幾步可以跳過

| 你們的步驟 | FreeSurfer 對應 | 能跳過嗎 |
|---|---|---|
| ① N4 bias correction | `nu.mgz`（N4，`AntsN4BiasFieldCorrectionFs`）| ✅ **已做過** |
| ② 去顱骨 | `norm.mgz` / `brain.mgz` **都已遮罩過**（watershed + atlas）| ✅ **已做過** → 用 `--no-brain-extract` |
| ③ Affine 對位到 atlas | ⚠️ 見下方「空間不一致」 | ❌ **不能跳** |
| ④ 正規化到 [0,1] | FreeSurfer 是「白質=110 的 uchar」，不是 [0,1] | ❌ **要做** |
| ⑤ 打包 npz | — | ❌ 要做 |

**你們的腳本沒有跳過 N4 的旗標。** 不過對已經 N4 過的影像再跑一次 N4，
結果幾乎不變（只是多花時間），所以**不改程式也能用**；想省時間再加旗標即可。

---

## 3. ⚠️ 空間不一致：MNI305 vs MNI152 09c

**這是最容易踩雷的一點。**

- FreeSurfer 的 `talairach.xfm` 對位目標是 **MNI305**（`mni305.cor.mgz`，305 顆腦平均）
- 你們的 atlas 是 **MNI152 09c**（`atlas_mni152_09c_v3.nii.gz`）

**兩者不是同一個空間。** 所以：

> ❌ **不要**直接套用 `talairach.xfm` 就以為對齊到你們的 atlas 了。
> ✅ **繼續用你們現有的 `ants.registration(...)`** 把 FreeSurfer 的 `norm.mgz` 對到你們的 atlas。

好處：你們的 atlas、shape (192,224,192)、正規化方式**全部不用改**，跟現有 IXI 資料完全相容，
可以混著訓練或直接沿用既有模型。

---

## 4. ⭐ 最大加值：標籤（可以算 Dice 了）

你們現在的 npz **只有 `vol`**。配準模型評估「對得準不準」的標準做法是
**標籤傳播 + Dice**：把 atlas 的解剖標籤用形變場搬到病人空間，再跟病人自己的標籤比重疊。
沒有標籤就只能看損失函數，**沒辦法報 Dice**。

FreeSurfer 的 `aseg.mgz` 正好補上這塊。

> 補充：VoxelMorph / TransMorph 論文用的標籤**就是 FreeSurfer 做的**。
> TransMorph 在 IXI 上評估用 **29 個解剖結構**（論文正文；其 repo 的 `dice_val_VOI` 是 30 個 label index）。
> 所以你們加上 `aseg` 之後，評估方式就跟論文對齊了。

### ⚠️ 標籤處理的三個鐵則

1. **必須用「跟影像完全相同」的變換** —— 不能各自對位一次
2. **內插必須用最近鄰（nearestNeighbor）** —— 用 linear 會在標籤 17 和 10 之間插出 13.5 這種不存在的值
3. **標籤不要正規化、不要轉 float** —— 保持整數（`int16` 即可）

⚠️ 另外：aseg 的標籤值是 FreeSurfer LUT 編號（2, 3, 4, 7, 8, 10, 11, … 41, 42 …），
**不連續**。算 Dice 時要自己挑一組結構（可參考 TransMorph repo 的 `Anatomical_Structures.md`）。

---

## 5. 具體怎麼改（改動很小）

在 `preprocess_ixi.py` 的第 4 步之後插入標籤處理即可：

```python
# 4. 對位到 atlas（Affine）—— 你們原本就有
reg = ants.registration(
    fixed  = atlas_ants,
    moving = img_brain,
    type_of_transform = 'Affine',
    verbose = False,
)
img_reg = reg['warpedmovout']

# ── 4b. 新增：用「同一個變換」把 aseg 標籤搬到 atlas 空間 ──
seg_reg_np = None
if seg_path is not None:                      # seg_path = 對應的 aseg .nii.gz
    seg = ants.image_read(seg_path)
    seg_reg = ants.apply_transforms(
        fixed         = atlas_ants,
        moving        = seg,
        transformlist = reg['fwdtransforms'],   # ⭐ 重用影像的變換
        interpolator  = 'nearestNeighbor',      # ⭐ 標籤必須最近鄰
    )
    seg_reg_np = seg_reg.numpy().astype(np.int16)   # 不正規化、保持整數

# 5~6. 影像 shape 檢查 + 正規化 —— 你們原本就有，不用改

# 7. 打包（多存一個 key）
if seg_reg_np is not None:
    np.savez_compressed(dst_path, vol=img_np, seg=seg_reg_np)
else:
    np.savez_compressed(dst_path, vol=img_np)
```

> `seg` 這個 key 名稱跟 VoxelMorph 官方 OASIS 資料的慣例一致，之後要接官方範例比較順。

> ✅ **2026-08-23 更新：上面這段程式碼已由 VoxelMorph 端實測通過**（ANTsPy 0.6.3，
> `ASD/verify_seg_transform.py`）。驗證方法：用同一個 `reg['fwdtransforms']` 把影像與標籤
> 都以 nearestNeighbor 搬過去，再把搬完的影像用原門檻重新量化、與搬完的標籤逐 voxel 比對
> —— 變換若真的共用則必然 bit-exact。結果 **0 / 8,257,536 不一致**。
> 另有反證：同一組標籤改用 linear 會捏造出 1,536,912 個原標籤集裡沒有的值。
> → 本文件原先「此段未實測」的警語**已不再適用**。

### FreeSurfer 那邊先把檔案轉出來

（在 Ubuntu VM 上跑，FreeSurfer 環境）

```bash
export SUBJECTS_DIR=/home/cheng/workspace/subjects/ASD_result
OUT=/home/cheng/workspace/fs_for_vxm
mkdir -p $OUT/norm $OUT/aseg

for d in $SUBJECTS_DIR/*/; do
  s=$(basename "$d")
  grep -q "finished without error" "$d/scripts/recon-all.log" 2>/dev/null || continue
  mri_convert "$d/mri/norm.mgz" "$OUT/norm/${s}.nii.gz"
  mri_convert "$d/mri/aseg.mgz"  "$OUT/aseg/${s}.nii.gz"  -rt nearest
done
```
> 只轉「跑成功」的；`-rt nearest` 是保險（這裡沒有重取樣，其實不會內插）。

### 為什麼是 `norm.mgz` 不是 `brain.mgz`（2026-08-23 決定）

| 檔案 | 怎麼來的 |
|---|---|
| `norm.mgz` | `mri_ca_normalize` 的產物：用 **GCA 圖譜**當控制點做亮度正規化，再遮罩 |
| `brain.mgz` | 再多做一次 `mri_normalize -aseg`，改用**該受試者自己的 aseg** 當控制點 |

也就是 `brain` 比 `norm` 多一道、而且更貼身的正規化。
FreeSurfer 端原本推薦 `brain`（跨受試者亮度更一致）；**使用者選 `norm`**
（在 VoxelMorph 生態裡更常見，少一道處理、離原始更近）。

> **兩者都是合理選擇，目前沒有文獻依據說哪個對配準訓練更好。**
> 重點是**全體一致**——167 顆全是 `norm.mgz`，沒有混用。
> 📌 要發表的話，方法學要寫明用的是 `norm.mgz`；
> 若配準結果不理想，這是該回頭檢視的變因之一。
> 影像也可以改用 `norm.mgz`（同樣去顱骨+正規化）——**兩者都可以，重點是全體一致**。

---

## 6. ⚠️⚠️ 資料切分：有 data leakage 風險

你們現在的切分是**檔案層級**的隨機 shuffle：

```python
random.shuffle(all_files)
n_train = int(n * 0.90)
```

**FreeSurfer 這批資料裡有疑似「同一個人的多次掃描」**：

```
A016_1 / A016_2
A0131  / A0132
```

如果一個進 train、一個進 test，模型等於**看過答案**，測出來的 Dice 會虛高。

> ✅ **切分要以「受試者」為單位，不是以「檔案」為單位。**

**⚠️ 更正（2026-08-23）：`A013` 本身也存在**，所以那組是 **A013 + A0131 + A0132 三個**，不是兩個。
這讓命名更曖昧：`A0131` 可能是「A013 的第 1 次」，也可能是一個獨立受試者。
**字串規則判斷不出來，不要用猜的規則寫死在 code 裡。**

✅ VoxelMorph 端已對全部 167 個 ID 掃描過（偵測「只差結尾一個數字」與「某 ID 是另一 ID 的前綴」兩種型態），
結論：**這種曖昧全批只有 `A013/A0131/A0132` 這一組，沒有第二處**。
→ 所以要跟老師確認的只有這一件事，不需要逐一比對 167 個。

目前用命名規則能看到的疑似同組（**未經確認**）：
- `A013`：A013, A0131, A0132　←（唯一曖昧處，待老師確認）
- `A016`：A016_1, A016_2

歸戶已做成外部設定檔（`--group-map`，TSV），確認後改設定檔即可，不用改程式。

---

## 7. ⚠️ 資料品質：這幾顆不要用

FreeSurfer 端已做過品質檢查，發現：

| 受試者 | 問題 | 處置 |
|---|---|---|
| **A043** | 影像雜訊過高、灰白對比不足（白質只認出約 5%）| 🔴 **排除** |
| **T085** | 只有 120/192 張切片（缺 0001–0072）| 🔴 **排除** |
| **A012** | 資料夾混了兩次不同掃描（0801 + A012 各 192 張）| 🟡 修好前**不要用** |
| **T065** | 資料夾名 T065，但 DICOM 病人 ID 是 **T056** | 🟠 身分待確認，**先不要納入** |

> 完整診斷在 `D:\MyHome\MRI\FreeSurfer\docs\ASD_資料品質記錄.md`。

### ✅ 混掃描全面檢查：2026-08-23 已完成，通過

原本擔心的是：A012 是因為疊成 384 層超過 FreeSurfer 的 256 上限才報錯露餡；
**若某顆混了但總層數沒超標，recon-all 不會報錯，會安靜跑出一顆「兩個人疊在一起」的腦**。

檢查結果：**167 個資料夾、異常 0**，每個資料夾恰好 192 張 .IMA 且只含單一病人 ID，
167 × 192 = **32,064** 與實際檔案總數完全一致（數字閉合）。
同時反證了 A012 的修復成功、沒有其他資料夾混掃描、沒有缺切片的。

> ⚠️ 殘餘風險：本次驗證的是「檔案數 + 病人 ID 一致性」，**未逐檔驗證檔案完整性**。
> 但 167 顆全部通過 `recon-all -all`（需能完整讀取整個序列才可能成功），視為足夠強的旁證。

> 📌 上表的 **A012 已於 2026-08-23 修復並重跑完成，已納入最終清單**（167 顆）。
> A043 / T085 / T065 維持排除。

---

## 7b. 🔒 執行閘門（VoxelMorph 端已實作）

批次前處理腳本預設**拒絕執行**（exit code 2），只允許 `--dry-run` 與 `--only <SUBJECT>`；
必須加 `--list-is-final` 才會跑全批。

🔓 **2026-08-23：閘門條件已滿足。** 混掃描檢查通過，FreeSurfer 端已交付標明 FINAL 的
清單（`ASD_可用清單_FINAL.txt`，167 個）並明確授權解除。機制保留給之後的新資料集用。

---

## 8. 建議執行順序

1. ~~等 FreeSurfer 端完成「混掃描」全面檢查~~ ✅ **已完成**，FINAL 清單 167 顆
2. ~~用第 5 節的 bash 把 `norm.mgz` / `aseg.mgz` 轉成 `.nii.gz`~~ ✅ **已完成**（285 MB，但在**另一台機器**上，仍待跨機器搬運）
3. **先拿 1 顆**跑 `ASD\preprocess_fs.py --only <SUBJECT> --save-nii`，
   用 `nii/` 的輸出在 ITK-SNAP／Freeview 確認：
   - 影像有對到 atlas 空間、shape = (192, 224, 192)
   - **標籤和影像完全疊合**（這是最重要的檢查）
   - 標籤值仍是整數、沒有出現奇怪的中間值
4. 確認無誤 → 批次跑全部
5. 切分改成**受試者層級**
6. 訓練；評估時用 `seg` 算 Dice（記得同時報 Jacobian 折疊率 |J|≤0，
   只看 Dice 會被「亂折疊硬湊高分」騙）

---

## 9. 誠實邊界

- ✅ **已實際查證**：你們的 `preprocess_ixi.py` 流程、npz 的 keys 與 shape、
  FreeSurfer 各產物的內容與空間規格、上述四顆受試者的品質問題（有 log／檔案證據）。
- ✅ **已由 VoxelMorph 端實測通過**（2026-08-23）：第 5 節的程式碼片段。
  原先標為「未實測」，現已驗證（見第 5 節的更新說明），此警語作廢。
- ⚠️ **未確認**：A016_1/A016_2、A0131/A0132 是否真為同一人；T065 的真實身分。
- 📌 `talairach.xfm`（→MNI305）在本流程中**用不到**，故意不使用——避免與你們的 MNI152 atlas 混淆。

---

## 10. 有問題找誰

FreeSurfer 那條線的細節（哪個檔是什麼、品質問題、要不要重跑某顆），
可以直接問跑 FreeSurfer 那邊的 session，或翻這些文件：

- `D:\MyHome\MRI\FreeSurfer\docs\ASD_資料品質記錄.md` — 哪幾顆壞掉、為什麼
- `D:\MyHome\MRI\FreeSurfer\FreeSurfer_Tutorials.md` — §2.1 conform／各產物是什麼
- `D:\MyHome\MRI\FreeSurfer\meeting\2026-07-25_meeting準備.md` — §5.2b 完整處理步驟表
