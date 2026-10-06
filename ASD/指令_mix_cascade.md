# mix_cascade 操作單（cmd 版）

> **這顆在問什麼**：改架構的第 1 步（CLAUDE.md 待辦 5）——**把兩顆 VoxelMorph 串起來**（RCN，ICCV 2019）。
> 第一顆先對，第二顆拿「對好一次的影像」專門修第一顆沒對好的地方，兩顆一起訓練。
> 其他全部跟 mix_exp6 一樣（速度場、全尺寸積分、平滑權重 1、同一份資料、250 epoch），**只差「串了兩顆」**。
>
> 為什麼值得跑（第 0 步，2026-10-06，不用訓練、把現成的模型連跑兩次）：
>
> | | 跑 1 次 | 同一顆連跑 2 次 |
> |---|---|---|
> | mix_exp6（預設寬度）| 0.8051 | **0.8136**（51 位全部變好）|
> | mix_wide_vel（加寬 2 倍）| 0.8111 | 0.8188 |
>
> 同一顆沒學過「修正」都能進步，真的訓練一顆專門修的第二顆，應該至少有這麼多。
>
> 跑的機器：**AI**，**等 mix_exp8／9 跑完再開始**（同一張顯卡，不能同時跑）。指令全部是 **cmd（命令提示字元）** 語法。
> 老師如果不希望改架構，這顆就不用跑。
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

確認拉到新程式（兩個檔都要有）：

```bat
dir ASD\arch.py ASD\train_arch.py
```

確認 mix_exp8／9 已經跑完、顯卡空出來了再開始。
需要 `data\mixed_preprocessed_v2\`（train 418 / val 51 / test 51），跟 mix_exp6 同一份，AI 上已經有了。

---

## 1. 起跑前檢查（不會真的開始訓練）

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --gpu 0 --check-only
```

要看到這一行：`架構 : 串 2 顆 VoxelMorph（RCN；ASD/train_arch.py），每一顆都罰平滑`。

**顯存**：2026-10-06 在筆電上用小尺寸（96×112×96，全尺寸的 1/8）實測、乘 8 外插：

| 設定 | 實際用到 | PyTorch 想預留 |
|---|---|---|
| 1 顆（＝ mix_exp6）| 8.8 GB（之前量的是 8.7 GB，對得上）| 15.4 GB |
| **串 2 顆（這顆）** | **16.9 GB** | **約 23.5 GB** |

AI 是 24 GB：真正用到的放得下，但 PyTorch 想預留的快滿了，**一定要先打第 2 步的 `set` 那行**。
⚠️ `--check-only` 只會印出卡有多大，不會真的建模型試跑；警告門檻 7.5 GB 是給預設的，**看到警告不用理**。

---

## 2. 訓練

🔴 **先打這一行**（限制 PyTorch 最多只佔顯卡的 85%，跟 mix_wide_vel 那次一樣）：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85
```

- 只對**這個 cmd 視窗**有效。**換新視窗（例如中斷後要續跑）要再打一次**
- 等號前後**不要有空白**；確認：`echo %PYTORCH_CUDA_ALLOC_CONF%`

然後在**同一個視窗**開始訓練：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --gpu 0
```

包裝腳本會自己存 `log\mix_cascade.txt`、`log\mix_cascade_script.txt`、`models\mix_cascade\`，**不用自己加 `>` 導向**。

**要跑多久**（看第 2 步以後的 `time:`）：

| | 每步 | 250 epoch |
|---|---|---|
| **預期** | **約 3.5～4 秒**（一顆在 AI 上 1.84 秒，兩顆約 2 倍）| **約 25～28 小時** |

| 看到 | 怎麼辦 |
|---|---|
| 噴 `CUDA out of memory` | 把 `set` 那行的 `0.85` 改成 `0.9` 重打，再跑下面的續跑指令 |
| `0.9` 還是 OOM | 續跑指令最後再加 `--grad-checkpoint`（省顯存模式：數字一樣、顯存約一半、慢約 1.24 倍；之後續跑也要一直加著）|
| 每步**超過 8 秒** | 多半是 `set` 沒生效（溢位到一般記憶體）。`Ctrl + C` 停掉，`echo` 檢查、重打 `set`，再續跑 |
| 還是不對 | 先停下來跟我說，**不要自己改參數** |

**中斷了怎麼辦**：新視窗先照第 0 步 `cd`、啟動虛擬環境，**再打一次 `set` 那行**，然後：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --gpu 0 --resume
```

---

## 3. 訓練完 → 用驗證集挑 epoch

🔴 **不要拿 test 挑 epoch。** test 只能跑一次。（評估程式已經讀得懂串接的模型，不用另外處理）

```bat
python ASD\test_dice.py --model-dir models\mix_cascade --test-dir data\mixed_preprocessed_v2\val --exp-name mix_cascade --step 5 --gpu 0
```

接著把起點複製過來、畫圖（四行可以一起貼）：

```bat
copy models\mix_exp2\dice_baseline.csv models\mix_cascade\
copy models\mix_exp2\dice_baseline_val.csv models\mix_cascade\
python ASD\plot_dice_curve.py --model-dir models\mix_cascade --label "串兩顆"
python ASD\plot_loss_curve.py --logs log\mix_cascade.txt log\mix_exp6.txt log\mix_wide_vel.txt --labels 串兩顆 一顆 加寬速度場 --out models\mix_cascade\loss_cascade_exp6_widevel.png
```

- 🔴 **不要加 `--baseline 0.6817`**（那是給 test 的）；最後一行**一定要 `--out`**
- ⚠️ 這顆 log 裡的平滑項是**兩顆加起來**，不能直接跟一顆的比；影像項可以比
- ⚠️ `log\mix_exp6.txt`、`log\mix_wide_vel.txt` 不在 AI 上的話，最後一行先跳過，帶回來我這邊畫

**挑法**：取 `dice_mean` 最大的那個 epoch。

---

## 4. test 評估（只跑一次）

把 `NNNN` 換成第 3 步挑出來的 epoch：

```bat
python ASD\test_dice.py --model models\mix_cascade\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_cascade --gpu 0
```

---

## 5. 要比的數字、怎麼判讀

| | mix_exp6 | 第 0 步：mix_exp6 連跑 2 次 | **mix_cascade** | mix_wide_vel |
|---|---|---|---|---|
| 網路 | 1 顆 | 1 顆用 2 次（沒訓練過修正）| **2 顆一起訓練** | 1 顆加寬 2 倍 |
| 參數 | 30 萬 | 30 萬 | 60 萬 | 120 萬 |
| test Dice | 0.8051 | 0.8136 | **？** | 0.8111 |
| 擠爆（每人幾點）| 0.1 | 43 | **？** | 0.1 |

👉 **跟第 0 步比（0.8136）**：
- **更高** → 第二顆真的學會「修」，比同一顆硬跑兩次好 → 串接值得，之後可以試串 3 顆、或跟加寬疊加
- **差不多** → 訓練沒多帶來什麼，好處主要來自「多跑一次」
- **明顯比較低** → 不對勁，先停下來跟我說

👉 **擠爆**：第 0 步連跑兩次每人 43 點；這顆兩段一起罰平滑，預期更少。

👉 **跟 mix_wide_vel 比**：參數只有一半，看「串起來」是不是比「加寬」划算（RCN 論文裡串兩顆贏過把一顆加深）。

帶回來之後逐人配對（51 位）我來算。

---

## 6. 視覺化（挑好 epoch 之後）

跟之前幾顆同樣四位，兩行都要把 `NNNN` 換掉：

```bat
for %s in (T054 D031 VNT045 sub-0043) do python ASD\visualize_dice.py --model models\mix_cascade\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_cascade\vis_%s --gpu 0
for %s in (T054 D031 VNT045 sub-0043) do python draw-img\visualize_reg_ixi.py --model models\mix_cascade\NNNN.pt --atlas IXI\atlas_mni152_09c_v3.npz --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_cascade\vis_%s --gpu 0
```

- `%s` 是 cmd 的迴圈變數，直接貼上就好（存成 `.bat` 才要改 `%%s`）
- 這步也可以不做，把 `.pt` 帶回來我這邊跑

---

## 7. 收尾

挑好 epoch、test 跑完之後，只留最佳那顆 `.pt`，其他刪掉（每個 2.3 MB，250 個約 580 MB）。

要帶回主機的東西：

```
models\mix_cascade\NNNN.pt                最佳 epoch 的權重
models\mix_cascade\dice_curve_val.csv     挑 epoch 的依據
models\mix_cascade\dice_NNNN.csv          test 結果
models\mix_cascade\*.png                  曲線圖
models\mix_cascade\vis_*\                 視覺化（有做的話）
log\mix_cascade.txt                       訓練 stdout
log\mix_cascade_script.txt                當初下的指令
```

---

## 注意事項

- 🔴 **訓練前先打 `set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85`**，換視窗要重打。只管記憶體、不改計算
- 🔴 **`--arch cascade --n-cascades 2`**：這顆跟 mix_exp6 唯一的差別
- 🔴 **`--int-steps 7 --int-downsize 1 --lambda 1.0`**：跟 mix_exp6 一樣（每一顆都用平滑權重 1）
- 🔴 **`--image-loss ncc`** 要明寫（預設是 mse）
- **不要加 `--seg-weight`**：跟 `--arch` 還不能一起用（包裝腳本會擋）
- 📌 跑完把結果補進 `ASD\ASD相關手冊.md` §25 和 `models\README.md`
