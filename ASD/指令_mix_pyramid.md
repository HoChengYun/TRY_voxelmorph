# mix_pyramid 操作單（cmd 版）

> **這顆在問什麼**：改架構的第 2 步（CLAUDE.md 待辦 5）——**由粗到細**。
> VoxelMorph 原文的 U-Net 特徵有粗有細，但形變只在最後輸出一次。這顆改成：
> - 解碼器從 1/16 開始，**每一層都出一個形變**；下一層先把移動影像的特徵照目前的形變**拉過去**，看到的是已經對好大半的樣子，只要修細節
> - 為了只拉「移動影像」的特徵，兩張影像改成**各自**用同一個編碼器抽特徵（由粗到細一定要這樣，兩件事綁在一起）
>
> 其他全部跟 mix_exp6 一樣（速度場、全尺寸積分、平滑權重 1、同一份資料、250 epoch），**只差架構**。
> 層數、通道數、卷積方塊都照 VoxelMorph 原本的；參數 41 萬（原本 30 萬，多的是解碼器每層多吃一份 atlas 的特徵）。
>
> 為什麼值得跑：「Mamba? Catch the Hype」那篇在 LPBA 上，VoxelMorph 67.0 → 加上「兩張各自抽特徵＋由粗到細」70.4，
> 是四種對位專用設計裡效果最大的；LUMIR 2024 第 1 名 SITReg、VFA 也都是由粗到細。
>
> 跑的機器：**AI**。**一次只跑一顆**：等 mix_exp8／9、mix_cascade 跑完（或你決定先跑這顆也可以，它比較快）。
> 指令全部是 **cmd（命令提示字元）** 語法。老師如果不希望改架構，這顆就不用跑。
> 最後更新：2026-10-06

---

## cmd 要注意的三件事

1. **每個指令都是一整行**：整行複製、貼上、按 Enter。看起來換行的地方是畫面自動折行，不是真的斷開。
2. **換磁碟要用 `cd /d`**：只打 `cd D:\...` 的話，如果目前在 C 槽，會停在原地不動
3. 虛擬環境啟動成功的話，提示字元前面會多一個 `(vxm_env)`

---

## 0. 開始前

```bat
cd /d D:\chengyun\TRY_voxelmorph
vxm_env\Scripts\activate.bat
git pull
```

確認拉到新程式：

```bat
dir ASD\arch.py ASD\train_arch.py ASD\verify_pyramid.py
```

確認顯卡空出來了（前一顆跑完）再開始。需要 `data\mixed_preprocessed_v2\`，跟 mix_exp6 同一份，AI 上已經有了。

---

## 1. 起跑前檢查（不會真的開始訓練）

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_pyramid --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch pyramid --gpu 0 --check-only
```

要看到這一行：`架構 : 由粗到細（兩張影像各自抽特徵、每一層都出形變；ASD/train_arch.py），每一層都罰平滑`。

**顯存**：2026-10-06 在筆電上用小尺寸實測、外插到全尺寸：

| 設定 | 實際用到 | PyTorch 想預留 |
|---|---|---|
| 1 顆（＝ mix_exp6）| 8.8 GB | 15.4 GB |
| **由粗到細（這顆）** | **9.7 GB** | **約 16.7 GB** |
| （參考）串兩顆 mix_cascade | 16.9 GB | 約 23.5 GB |

AI 24 GB 放得下。⚠️ `--check-only` 只會印出卡有多大，不會真的建模型試跑；看到 7.5 GB 的警告不用理。

---

## 2. 訓練

先打這一行（跟之前一樣，只管記憶體、不改計算；這顆不打應該也放得下，打了沒有壞處）：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85
```

然後在**同一個視窗**開始訓練：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_pyramid --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch pyramid --gpu 0
```

包裝腳本會自己存 `log\mix_pyramid.txt`、`log\mix_pyramid_script.txt`、`models\mix_pyramid\`，**不用自己加 `>` 導向**。

**要跑多久**（看第 2 步以後的 `time:`）：

| | 每步 | 250 epoch |
|---|---|---|
| **預期** | **約 2.7 秒**（一顆在 AI 上 1.84 秒 × 筆電實測的 1.49 倍）| **約 19 小時** |

| 看到 | 怎麼辦 |
|---|---|
| 噴 `CUDA out of memory` | 把 `set` 那行的 `0.85` 改成 `0.9` 重打，再跑下面的續跑指令 |
| 每步**超過 6 秒** | 不對勁：`Ctrl + C` 停掉，`echo %PYTORCH_CUDA_ALLOC_CONF%` 檢查，先跟我說，**不要自己改參數** |

**中斷了怎麼辦**：新視窗先照第 0 步 `cd`、啟動虛擬環境，再打一次 `set` 那行，然後：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_pyramid --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch pyramid --gpu 0 --resume
```

---

## 3. 訓練完 → 用驗證集挑 epoch

🔴 **不要拿 test 挑 epoch。** test 只能跑一次。（評估程式已經讀得懂這種模型，不用另外處理）

```bat
python ASD\test_dice.py --model-dir models\mix_pyramid --test-dir data\mixed_preprocessed_v2\val --exp-name mix_pyramid --step 5 --gpu 0
```

接著把起點複製過來、畫圖（四行可以一起貼）：

```bat
copy models\mix_exp2\dice_baseline.csv models\mix_pyramid\
copy models\mix_exp2\dice_baseline_val.csv models\mix_pyramid\
python ASD\plot_dice_curve.py --model-dir models\mix_pyramid --label "由粗到細"
python ASD\plot_loss_curve.py --logs log\mix_pyramid.txt log\mix_exp6.txt log\mix_wide_vel.txt --labels 由粗到細 一顆 加寬速度場 --out models\mix_pyramid\loss_pyramid_exp6_widevel.png
```

- 🔴 **不要加 `--baseline 0.6817`**（那是給 test 的）；最後一行**一定要 `--out`**
- ⚠️ 這顆 log 裡的平滑項是**五層加起來**，不能直接跟一顆的比；影像項可以比
- ⚠️ `log\mix_exp6.txt`、`log\mix_wide_vel.txt` 不在 AI 上的話，最後一行先跳過，帶回來我這邊畫

**挑法**：取 `dice_mean` 最大的那個 epoch。

---

## 4. test 評估（只跑一次）

把 `NNNN` 換成第 3 步挑出來的 epoch：

```bat
python ASD\test_dice.py --model models\mix_pyramid\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_pyramid --gpu 0
```

---

## 5. 要比的數字、怎麼判讀

| | mix_exp6 | mix_cascade（第 1 步）| **mix_pyramid** | mix_wide_vel |
|---|---|---|---|---|
| 改了什麼 | 原本的 VoxelMorph | 串兩顆 | **由粗到細** | 加寬 2 倍 |
| 參數 | 30 萬 | 60 萬 | 41 萬 | 120 萬 |
| test Dice | 0.8051 | （跑完補）| **？** | 0.8111 |
| 擠爆（每人幾點）| 0.1 | （跑完補）| **？** | 0.1 |

👉 **跟 mix_exp6 比**：只差架構，差多少就是「由粗到細」的貢獻。
👉 **跟 mix_wide_vel 比**：參數只有 1/3，看「改結構」是不是比「加寬」划算。
👉 **跟 mix_cascade 比**：兩種對位設計哪個比較有用；之後也可以兩個疊在一起。
👉 **擠爆**：每一層都是速度場（各自積分），預期接近 0。

帶回來之後逐人配對（51 位）我來算。

---

## 6. 視覺化（挑好 epoch 之後）

跟之前幾顆同樣四位，兩行都要把 `NNNN` 換掉：

```bat
for %s in (T054 D031 VNT045 sub-0043) do python ASD\visualize_dice.py --model models\mix_pyramid\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_pyramid\vis_%s --gpu 0
for %s in (T054 D031 VNT045 sub-0043) do python draw-img\visualize_reg_ixi.py --model models\mix_pyramid\NNNN.pt --atlas IXI\atlas_mni152_09c_v3.npz --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_pyramid\vis_%s --gpu 0
```

- `%s` 是 cmd 的迴圈變數，直接貼上就好（存成 `.bat` 才要改 `%%s`）
- 這步也可以不做，把 `.pt` 帶回來我這邊跑

---

## 7. 收尾

挑好 epoch、test 跑完之後，只留最佳那顆 `.pt`，其他刪掉（每個 1.6 MB，250 個約 390 MB）。

要帶回主機的東西：

```
models\mix_pyramid\NNNN.pt                最佳 epoch 的權重
models\mix_pyramid\dice_curve_val.csv     挑 epoch 的依據
models\mix_pyramid\dice_NNNN.csv          test 結果
models\mix_pyramid\*.png                  曲線圖
models\mix_pyramid\vis_*\                 視覺化（有做的話）
log\mix_pyramid.txt                       訓練 stdout
log\mix_pyramid_script.txt                當初下的指令
```

---

## 注意事項

- 🔴 **`--arch pyramid`**：這顆跟 mix_exp6 唯一的差別
- 🔴 **`--int-steps 7 --int-downsize 1 --lambda 1.0`**：跟 mix_exp6 一樣（每一層都用平滑權重 1）。`--int-downsize` 給 1 以外的值包裝腳本會擋
- 🔴 **`--image-loss ncc`** 要明寫（預設是 mse）
- **不要加 `--seg-weight`**：跟 `--arch` 還不能一起用（包裝腳本會擋）
- **不要跟其他訓練同時跑**（同一張卡）
- 📌 跑完把結果補進 `ASD\ASD相關手冊.md` §25 和 `models\README.md`
