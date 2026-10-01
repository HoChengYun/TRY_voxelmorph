# mix_exp6 操作單（cmd 版）

> **這顆在問什麼**：mix_exp5 證明了，同樣全尺寸、同樣平滑權重 2，**速度場比位移場好**
> （0.8026 vs 0.8005，而且擠爆 0%）。但 mix_exp3（位移場、平滑權重 1）還是最高（0.8062），
> 差在平滑權重。**速度場把平滑權重也降到 1，能不能追上 mix_exp3，又不擠爆？**
>
> **mix_exp6 ＝ 速度場、全尺寸積分（跟 mix_exp5 一樣），只把平滑權重從 2 降到 1。**
> 放進來之後，全尺寸的四顆剛好湊成 2 × 2：
>
> | 實驗 | 版本 | `--int-steps` | `--int-downsize` | `--lambda` | 實際平滑權重 | test Dice | 擠爆 |
> |---|---|---|---|---|---|---|---|
> | mix_exp5 | 速度場 | 7 | 1 | 2.0 | 2 | 0.8026 | 0.000% |
> | mix_exp4 | 位移場 | 0 | 1 | 2.0 | 2 | 0.8005 | 0.053% |
> | **mix_exp6** | **速度場** | **7** | **1** | **1.0** | **1** | **？** | **？** |
> | mix_exp3 | 位移場 | 0 | 1 | 1.0 | 1 | 0.8062 | 0.199% |
>
> - **mix_exp6 − mix_exp5 ＝ 只差平滑權重**（都是速度場）
> - **mix_exp6 − mix_exp3 ＝ 只差版本**（平滑權重都是 1）
>
> 跑的機器：**AI**（這台筆電 8 GB 跑不動）。指令全部是 **cmd（命令提示字元）** 語法。
> 最後更新：2026-09-30

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

需要 `data\mixed_preprocessed_v2\`（train 418 / val 51 / test 51），跟 mix_exp5 同一份，AI 上已經有了。

---

## 1. 起跑前檢查（不會真的開始訓練）

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp6 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --gpu 0 --check-only
```

確認它印出來的 atlas 是 `atlas_mni152_09c_v3.npz`。

**顯存**：跟 mix_exp5 一樣約 **8.7 GB**（網路和積分方式都一樣，λ 不影響顯存）。

🔴 **`--int-downsize` 一定要寫 1。** 它的預設是 2，漏寫的話會變成「一半解析度、平滑權重 2」，
等於把 mix_exp2 重跑一次。這顆最容易出錯的地方就是這裡。

🔴 **`--lambda` 是 1.0**（mix_exp5 是 2.0）。實際平滑權重 = λ × int-downsize = 1.0 × 1 = 1，
才跟 mix_exp3 對齊。

---

## 2. 訓練

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp6 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --gpu 0
```

包裝腳本會自動處理這些，**不用自己加 `>` 導向**：

| 它會做的事 | 位置 |
|---|---|
| 存 stdout | `log\mix_exp6.txt` |
| 存當初下的指令 | `log\mix_exp6_script.txt` |
| 存權重 | `models\mix_exp6\` |
| 帶 atlas | 預設就是 `IXI\atlas_mni152_09c_v3.npz` |

**要跑多久**：mix_exp5 在 AI 上實際是每步 1.84 秒、12.7 小時，這顆應該差不多。
以第一個 epoch 印出來的 `time:` 為準。

⚠️ 每步**超過 5 秒**就是顯存塞不下、在偷借系統記憶體。遇到的話先停下來跟我說，**不要自己改參數**。

**中斷了怎麼辦**：用下面這行，會從最後一個 `.pt` 接著跑（就是上面那行最後加 `--resume`）：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp6 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --gpu 0 --resume
```

---

## 3. 訓練完 → 用驗證集挑 epoch

🔴 **不要拿 test 挑 epoch。** test 只能跑一次。

```bat
python ASD\test_dice.py --model-dir models\mix_exp6 --test-dir data\mixed_preprocessed_v2\val --exp-name mix_exp6 --step 5 --gpu 0
```

輸出 `models\mix_exp6\dice_curve_val.csv`。接著把起點複製過來、畫圖（四行可以一起貼）：

```bat
copy models\mix_exp2\dice_baseline.csv models\mix_exp6\
copy models\mix_exp2\dice_baseline_val.csv models\mix_exp6\
python ASD\plot_dice_curve.py --model-dir models\mix_exp6 --label "速度場 全尺寸 平滑權重1"
python ASD\plot_loss_curve.py --logs log\mix_exp6.txt log\mix_exp5.txt log\mix_exp3.txt --labels 速度場權重1 速度場權重2 位移場權重1 --out models\mix_exp6\loss_exp5_exp6_exp3.png
```

- 起點跟模型無關，mix_exp2 那兩份直接用。`plot_dice_curve.py` 會自己讀：test 起點 0.6882、驗證集起點 0.6817
- 🔴 **不要加 `--baseline 0.6817`**：那個參數是給 test 的，會把驗證集的起點標成 test
- 🔴 最後一行**一定要 `--out`**，同時畫多份 log 沒給會直接停
- ⚠️ `log\mix_exp5.txt`、`log\mix_exp3.txt` 要在 AI 上。沒有的話最後一行先跳過，帶回來我這邊畫

**挑法**：取 `dice_mean` 最大的那個 epoch。相鄰 epoch 的抖動約 ±0.003，曲線平的時候別太在意差 0.001 的名次。

---

## 4. test 評估（只跑一次）

把 `NNNN` 換成第 3 步挑出來的 epoch（例如 `0150`）：

```bat
python ASD\test_dice.py --model models\mix_exp6\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_exp6 --gpu 0
```

輸出 `models\mix_exp6\dice_NNNN.csv`（逐人、逐結構）。

---

## 5. 要比的數字、怎麼判讀

| | mix_exp5 | **mix_exp6** | mix_exp3 |
|---|---|---|---|
| 版本 | 速度場 | 速度場 | 位移場 |
| 平滑權重 | 2 | 1 | 1 |
| test Dice | 0.8026 | **？** | 0.8062 |
| 擠爆比例 | 0.000% | **？** | 0.199% |

👉 **Dice**：
- **比 mix_exp5 高** → 平滑權重降到 1，速度場也吃得到好處
- **追上或超過 mix_exp3（0.806）** → 速度場＋全尺寸是目前最好的設定（不加寬的話）
- **比 mix_exp5 還低** → 速度場不適合放鬆，平滑權重 2 就是它的甜蜜點

👉 **擠爆比例**：
- **還是接近 0%** → 速度場在平滑權重 1 也不擠爆，同時贏過 mix_exp3 的擠爆
- **明顯上升** → 速度場的「不擠爆」也有極限（IXI exp4 平滑權重 0.01 時到 2.26%）

帶回來之後逐人配對（51 位）我來算。

---

## 6. 視覺化（挑好 epoch 之後）

跟之前幾顆同樣四位。兩行都要把 `NNNN` 換掉，可以一起貼：

```bat
for %s in (T054 D031 VNT045 sub-0043) do python ASD\visualize_dice.py --model models\mix_exp6\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_exp6\vis_%s --gpu 0
for %s in (T054 D031 VNT045 sub-0043) do python draw-img\visualize_reg_ixi.py --model models\mix_exp6\NNNN.pt --atlas IXI\atlas_mni152_09c_v3.npz --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_exp6\vis_%s --gpu 0
```

每位 8 張（第一行 3 張、第二行 5 張）。
- `%s` 是 cmd 的迴圈變數，直接貼上就好。⚠️ 如果改存成 `.bat` 檔再執行，要改成 `%%s`
- 📌 形變網格是新樣式（黃線、間距 6），跟 mix_exp5 同一種，可以直接並排比
- 這步也可以不做，把 `.pt` 帶回來我這邊跑

---

## 7. 收尾

挑好 epoch、test 跑完之後，只留最佳那顆 `.pt`，其他刪掉（250 個檔案很佔空間）。

要帶回主機的東西：

```
models\mix_exp6\NNNN.pt                最佳 epoch 的權重
models\mix_exp6\dice_curve_val.csv     挑 epoch 的依據
models\mix_exp6\dice_NNNN.csv          test 結果
models\mix_exp6\*.png                  曲線圖
models\mix_exp6\vis_*\                 視覺化（有做的話）
log\mix_exp6.txt                       訓練 stdout
log\mix_exp6_script.txt                當初下的指令
```

---

## 注意事項

- 🔴 **`--int-downsize 1`**：預設是 2，漏寫就等於重跑 mix_exp2
- 🔴 **`--lambda 1.0`**：這顆跟 mix_exp5 唯一的差別
- 🔴 **`--image-loss ncc`** 要明寫（預設是 mse）
- **`--atlas-seg` 不用換**：這是 FreeSurfer 那組，用預設的 `atlas_mni152_09c_v3_seg.npz`
- 其他全部跟 mix_exp5 一樣：同一份資料、U-Net 預設寬度、250 epoch、學習率 1e-4
- 📌 跑完把結果補進 `ASD\ASD相關手冊.md` §20.5 和 `models\README.md`
