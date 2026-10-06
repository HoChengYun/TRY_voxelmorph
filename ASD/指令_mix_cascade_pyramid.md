# mix_cascade_pyramid 操作單（cmd 版）

> **這顆在問什麼**：改架構的 2 × 2 最後一格——**串兩顆＋每一顆都是由粗到細**（兩個疊在一起）。
>
> | | 不串 | 串兩顆 |
> |---|---|---|
> | **原本的 VoxelMorph** | mix_exp6：0.8051 | mix_cascade（第 1 步）|
> | **由粗到細** | mix_pyramid（第 2 步）| **mix_cascade_pyramid（這顆）** |
>
> - 跟 mix_cascade 只差「每一顆換成由粗到細」；跟 mix_pyramid 只差「串了兩顆」
> - 看兩種設計的進步**加不加得起來**：加得起來＝疊在一起的進步約等於兩者相加；重疊＝兩個在補同一件事
>
> 🔴 **建議等 mix_cascade 和 mix_pyramid 都跑完、兩個都有進步再跑**。其中一個沒進步的話，疊在一起就沒意義了，這顆不用跑。
>
> 跑的機器：**AI**，一次只跑一顆。指令全部是 **cmd（命令提示字元）** 語法。老師如果不希望改架構，這顆就不用跑。
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
dir ASD\arch.py ASD\train_arch.py ASD\verify_pyramid.py
```

確認顯卡空出來了（前一顆跑完）再開始。需要 `data\mixed_preprocessed_v2\`，跟 mix_exp6 同一份。

---

## 1. 起跑前檢查（不會真的開始訓練）

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --gpu 0 --check-only
```

要看到：`架構 : 串 2 顆由粗到細的網路（兩個疊在一起；ASD/train_arch.py），每一顆的每一層都罰平滑`。

**顯存**（2026-10-06 在筆電上用小尺寸實測、外插到全尺寸）：

| 設定 | 實際用到 | PyTorch 想預留 |
|---|---|---|
| 1 顆（＝ mix_exp6）| 8.8 GB | 15.4 GB |
| **這顆** | **19.0 GB** | **約 26 GB** |
| **這顆＋省顯存模式**（`--grad-checkpoint`）| **9.7 GB** | 約 16 GB |

AI 是 24 GB：一般模式真正用到的放得下，但很滿，**要打第 2 步的 `set ...:0.9`**；萬一還是不夠，就改用省顯存模式（第 2 步最後）。
⚠️ 看到 `--check-only` 7.5 GB 的警告不用理。

---

## 2. 訓練

🔴 **先打這一行**（這顆比較大，用 **0.9**，不是之前的 0.85）：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.9
```

- 只對**這個 cmd 視窗**有效，換新視窗要再打一次；等號前後不要有空白；確認：`echo %PYTORCH_CUDA_ALLOC_CONF%`

然後在**同一個視窗**開始訓練：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --gpu 0
```

**要跑多久**（看第 2 步以後的 `time:`）：

| | 每步 | 250 epoch |
|---|---|---|
| **一般模式（預期）** | **約 5.3 秒**（一顆在 AI 上 1.84 秒 × 筆電實測 2.86 倍）| **約 37 小時** |
| 省顯存模式 | 約 6.5 秒（3.54 倍）| 約 45 小時 |

| 看到 | 怎麼辦 |
|---|---|
| 噴 `CUDA out of memory` | 改用**省顯存模式**（下面），從頭跑 |
| 每步**超過 10 秒** | 多半是 `set` 沒生效（溢位到一般記憶體）。`Ctrl + C` 停掉，`echo` 檢查；還是不行就改省顯存模式 |
| 還是不對 | 先停下來跟我說，**不要自己改參數** |

**省顯存模式**（OOM 才用）：每一顆的中間結果不留、反向傳播時再算一次。**算出來的數字一樣**（筆電實測梯度差 2e-7），只是慢約 1.24 倍。
新視窗的話先照第 0 步 `cd`、啟動虛擬環境，然後：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --grad-checkpoint --gpu 0 --resume --yes
```

（`--resume`：如果一般模式已經存了幾個 `.pt`，從最後一個接著跑；還沒存到的話從頭開始。`--yes`：不用回答確認問題）

**中斷了怎麼辦**：新視窗照第 0 步 `cd`、啟動虛擬環境、再打一次 `set` 那行，然後把**同一行訓練指令**後面加 `--resume`
（之前用省顯存模式的話，`--grad-checkpoint` 也要留著）。

---

## 3. 訓練完 → 用驗證集挑 epoch

🔴 **不要拿 test 挑 epoch。** test 只能跑一次。

```bat
python ASD\test_dice.py --model-dir models\mix_cascade_pyramid --test-dir data\mixed_preprocessed_v2\val --exp-name mix_cascade_pyramid --step 5 --gpu 0
```

```bat
copy models\mix_exp2\dice_baseline.csv models\mix_cascade_pyramid\
copy models\mix_exp2\dice_baseline_val.csv models\mix_cascade_pyramid\
python ASD\plot_dice_curve.py --model-dir models\mix_cascade_pyramid --label "疊在一起"
python ASD\plot_loss_curve.py --logs log\mix_cascade_pyramid.txt log\mix_cascade.txt log\mix_pyramid.txt log\mix_exp6.txt --labels 疊在一起 串兩顆 由粗到細 一顆 --out models\mix_cascade_pyramid\loss_2x2.png
```

- 🔴 **不要加 `--baseline 0.6817`**；最後一行**一定要 `--out`**
- ⚠️ log 裡的平滑項是**兩顆 × 五層加起來**，不能跟別顆比；影像項可以比
- ⚠️ 哪份 log 不在 AI 上，最後一行先跳過，帶回來我這邊畫

**挑法**：取 `dice_mean` 最大的那個 epoch。

---

## 4. test 評估（只跑一次）

```bat
python ASD\test_dice.py --model models\mix_cascade_pyramid\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_cascade_pyramid --gpu 0
```

---

## 5. 要比的數字、怎麼判讀

| | 不串 | 串兩顆 |
|---|---|---|
| **原本的 VoxelMorph** | mix_exp6：0.8051 | mix_cascade：（跑完補）|
| **由粗到細** | mix_pyramid：（跑完補）| **這顆：？** |

👉 **加不加得起來**：（這顆 − mix_exp6）跟（mix_cascade − mix_exp6）＋（mix_pyramid − mix_exp6）比
- 差不多 → 兩種設計各補各的，可以一起用
- 明顯比較小 → 兩個在補同一件事（第 0 步就看到「多跑一次」跟「加寬」幫到的是同一批人）

帶回來之後逐人配對（51 位）我來算。

---

## 6. 視覺化、收尾

跟 mix_cascade 的操作單一樣，把 `mix_cascade` 換成 `mix_cascade_pyramid`。
`.pt` 每個 3.1 MB，250 個約 780 MB，挑好之後只留最佳那顆。

要帶回主機的東西：`NNNN.pt`、`dice_curve_val.csv`、`dice_NNNN.csv`、`*.png`、`vis_*\`（有做的話）、`log\mix_cascade_pyramid.txt`、`log\mix_cascade_pyramid_script.txt`。

---

## 注意事項

- 🔴 **`--arch cascade --n-cascades 2 --stage pyramid`**：三個都要寫
- 🔴 **`--int-steps 7 --int-downsize 1 --lambda 1.0 --image-loss ncc`**：跟 mix_exp6 一樣
- 🔴 一般模式 `set ...:0.9`；省顯存模式 `set ...:0.85` ＋ `--grad-checkpoint`（中斷續跑時要一致）
- **不要加 `--seg-weight`**（跟 `--arch` 還不能一起用）；**不要跟其他訓練同時跑**
- 📌 跑完把結果補進 `ASD\ASD相關手冊.md` §25 和 `models\README.md`
