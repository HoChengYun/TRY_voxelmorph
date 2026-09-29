# mix_exp5 操作單（cmd 版）

> **這顆在問什麼**：簡報 P18「換成論文的版本 +0.003」，到底是
> **「速度場 → 位移場」**的功勞，還是**「半解析度 → 全解析度」**的功勞？
> 現在 mix_exp2 → mix_exp4 兩件事一起換了，分不開（手冊 §20.5 的 2026-09-29 更正）。
>
> **mix_exp5 ＝ 速度場（跟 mix_exp2 一樣積分 128 步），但改在全尺寸上積分。**
> 放進來之後，三顆兩兩只差一件事：
>
> | 實驗 | 版本 | 積分的解析度 | `--int-steps` | `--int-downsize` | `--lambda` | 實際平滑權重 |
> |---|---|---|---|---|---|---|
> | mix_exp2 | 速度場 | 一半 | 7 | 2 | 1.0 | 1.0 × 2 = **2** |
> | **mix_exp5** | **速度場** | **全尺寸** | **7** | **1** | **2.0** | 2.0 × 1 = **2** |
> | mix_exp4 | 位移場 | 全尺寸（不積分）| 0 | 1 | 2.0 | 2.0 × 1 = **2** |
>
> - **mix_exp5 − mix_exp2 ＝ 只差解析度**（都是速度場）
> - **mix_exp4 − mix_exp5 ＝ 只差版本**（都是全尺寸）
>
> 跑的機器：**AI**（這台筆電 8 GB 跑不動）。指令全部是 **cmd（命令提示字元）** 語法。
> 最後更新：2026-09-29（指令改成 cmd）

---

## cmd 要注意的三件事

1. **每個指令都是一整行**：整行複製、貼上、按 Enter。看起來換行的地方是畫面自動折行，不是真的斷開。
   （故意不用 `^` 接行：`^` 後面多一個空白就會斷掉，後半的參數沒吃到，有可能拿預設值安靜地跑下去）
2. **換磁碟要用 `cd /d`**：只打 `cd D:\...` 的話，如果目前在 C 槽，會停在原地不動
3. 虛擬環境啟動成功的話，提示字元前面會多一個 `(vxm_env)`

---

## 0. 開始前

```bat
cd /d D:\chengyun\TRY_voxelmorph
vxm_env\Scripts\activate.bat
git pull
```

⚠️ **`git pull` 一定要做**：2026-09-28 修了畫圖程式的中文字型，沒拉的話第 3 步圖上的中文標籤會變方框。

需要 `data\mixed_preprocessed_v2\`（train 418 / val 51 / test 51），跟 mix_exp2 / 3 / 4 / wide 同一份。
AI 那台之前跑 mix_wide 已經有了，不用再搬。

---

## 1. 起跑前檢查（不會真的開始訓練）

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp5 --image-loss ncc --lambda 2.0 --epochs 250 --int-steps 7 --int-downsize 1 --gpu 0 --check-only
```

確認它印出來的 atlas 是 `atlas_mni152_09c_v3.npz`。

**顯存**：實測約 **8.7 GB**（mix_exp3 是 7.3、mix_exp2 是 7.0；積分改在全尺寸做，多了約 1.5 GB）。
AI 那台跑過 13.5 GB 的 mix_wide，這顆沒問題。

🔴 **`--lambda` 一定要 2.0，不是 1.0。**
實際平滑權重 = λ × int-downsize。int-downsize 從 2 改成 1，λ 就要從 1.0 補成 2.0，
才能跟 mix_exp2（1.0 × 2）、mix_exp4（2.0 × 1）對齊。寫成 1.0 就又多一個變因，這顆就白跑了。

---

## 2. 訓練

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp5 --image-loss ncc --lambda 2.0 --epochs 250 --int-steps 7 --int-downsize 1 --gpu 0
```

包裝腳本會自動處理這些，**不用自己加 `>` 導向**：

| 它會做的事 | 位置 |
|---|---|
| 存 stdout | `log\mix_exp5.txt` |
| 存當初下的指令 | `log\mix_exp5_script.txt` |
| 存權重 | `models\mix_exp5\` |
| 帶 atlas | 預設就是 `IXI\atlas_mni152_09c_v3.npz` |

**要跑多久**：實測每步約是 mix_exp3 的 1.06 倍 → AI 上約 **1.6 秒/步、250 epoch 約 11 小時**。
以第一個 epoch 印出來的 `time:` 為準。

⚠️ 每步**超過 5 秒**就是顯存塞不下、在偷借系統記憶體（mix_wide 操作單第 1 步有說明）。
遇到的話先停下來跟我說，**不要自己改參數**：這顆任何一個參數動了，比較就不成立了。

**中斷了怎麼辦**：用下面這行，會從最後一個 `.pt` 接著跑（就是上面那行最後加 `--resume`）：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp5 --image-loss ncc --lambda 2.0 --epochs 250 --int-steps 7 --int-downsize 1 --gpu 0 --resume
```

---

## 3. 訓練完 → 用驗證集挑 epoch

🔴 **不要拿 test 挑 epoch。** test 只能跑一次。

```bat
python ASD\test_dice.py --model-dir models\mix_exp5 --test-dir data\mixed_preprocessed_v2\val --exp-name mix_exp5 --step 5 --gpu 0
```

輸出 `models\mix_exp5\dice_curve_val.csv`。接著把起點複製過來、畫圖（四行可以一起貼）：

```bat
copy models\mix_exp2\dice_baseline.csv models\mix_exp5\
copy models\mix_exp2\dice_baseline_val.csv models\mix_exp5\
python ASD\plot_dice_curve.py --model-dir models\mix_exp5 --label "速度場 全尺寸"
python ASD\plot_loss_curve.py --logs log\mix_exp5.txt log\mix_exp2.txt log\mix_exp4.txt --labels 速度場全尺寸 速度場半解析度 位移場全尺寸 --out models\mix_exp5\loss_exp2_exp5_exp4.png
```

- 起點跟模型無關，mix_exp2 那兩份直接用。`plot_dice_curve.py` 會自己讀：test 起點 0.6882、驗證集起點 0.6817
- 🔴 **不要加 `--baseline 0.6817`**：那個參數是給 test 的，會把驗證集的起點標成 test
- 🔴 最後一行**一定要 `--out`**，同時畫多份 log 沒給會直接停
- ⚠️ loss 圖的**平滑項三顆不能直接比大小**：速度場版的懲罰算在「積分前的速度」上（mix_exp2 還是縮小一半的速度），
  位移場版算在位移上。**看中間那格影像項就好**

**挑法**：取 `dice_mean` 最大的那個 epoch。相鄰 epoch 的抖動約 ±0.003，曲線平的時候別太在意差 0.001 的名次。

---

## 4. test 評估（只跑一次）

把 `NNNN` 換成第 3 步挑出來的 epoch（例如 `0240`）：

```bat
python ASD\test_dice.py --model models\mix_exp5\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_exp5 --gpu 0
```

輸出 `models\mix_exp5\dice_NNNN.csv`（逐人、逐結構）。

---

## 5. 要比的數字、怎麼判讀

| | mix_exp2 | **mix_exp5** | mix_exp4 |
|---|---|---|---|
| 版本 | 速度場 | 速度場 | 位移場 |
| 積分的解析度 | 一半 | 全尺寸 | 全尺寸 |
| test Dice | 0.7972 | **？** | 0.8005 |
| 擠爆比例 | 0.000% | **？** | 0.053% |

👉 **Dice 看 mix_exp5 落在哪**：
- **靠近 mix_exp4（0.800）** → 那 +0.003 主要是**解析度**的功勞，換成位移場本身沒差
- **靠近 mix_exp2（0.797）** → 主要是**版本**的功勞，解析度沒差
- **在中間** → 兩個都有，看各佔多少

👉 **擠爆比例**：速度場理論上不會擠爆，mix_exp5 預期還是接近 0%。
如果 exp5 是 0%、exp4 是 0.053%，就證明**擠爆是版本造成的，跟解析度無關**。

帶回來之後逐人配對（51 位）我來算，會看每一步有幾位變好。

---

## 6. 視覺化（挑好 epoch 之後）

跟 mix_exp2 / 3 / 4 / wide 同樣四位，才能並排比。兩行都要把 `NNNN` 換掉，可以一起貼：

```bat
for %s in (T054 D031 VNT045 sub-0043) do python ASD\visualize_dice.py --model models\mix_exp5\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_exp5\vis_%s --gpu 0
for %s in (T054 D031 VNT045 sub-0043) do python draw-img\visualize_reg_ixi.py --model models\mix_exp5\NNNN.pt --atlas IXI\atlas_mni152_09c_v3.npz --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_exp5\vis_%s --gpu 0
```

每位 8 張（第一行 3 張、第二行 5 張）。
- `%s` 是 cmd 的迴圈變數，直接貼上就好。⚠️ 如果改存成 `.bat` 檔再執行，要改成 `%%s`
- ⚠️ `visualize_reg_ixi.py` 不給 `--subject` 會隨機挑人，就對不起來了

---

## 7. 收尾

挑好 epoch、test 跑完之後，只留最佳那顆 `.pt`，其他刪掉（250 個檔案很佔空間）。

要帶回主機的東西：

```
models\mix_exp5\NNNN.pt                最佳 epoch 的權重
models\mix_exp5\dice_curve_val.csv     挑 epoch 的依據
models\mix_exp5\dice_NNNN.csv          test 結果
models\mix_exp5\*.png                  曲線圖
models\mix_exp5\vis_*\                 視覺化
log\mix_exp5.txt                       訓練 stdout
log\mix_exp5_script.txt                當初下的指令
```

---

## 注意事項

- 🔴 **`--lambda 2.0`**，理由見第 1 步。這是這顆最容易寫錯的地方
- 🔴 **`--image-loss ncc`** 要明寫（預設是 mse）
- **`--atlas-seg` 不用換**：這是 FreeSurfer 那組，用預設的 `atlas_mni152_09c_v3_seg.npz`
- 其他全部跟 mix_exp2 / 4 一樣：同一份資料、U-Net 預設寬度、250 epoch、學習率 1e-4
- 📌 跑完把結果補進 `ASD\ASD相關手冊.md` §20.5 和 `models\README.md`
