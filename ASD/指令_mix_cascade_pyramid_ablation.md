# 合併版為什麼變低（cmd 版）：第 1 顆 rep2 → 第 2 顆 只開 rcn → 第 3 顆 只開 dwp

> **目的**（手冊 §31.14，2026-11-05 使用者：「現在第一目標，是把 mix_cascade_pyramid_rcn_dwp 變低問題解決」）：
> 串兩顆＋由粗到細的合併版，舊的（不開開關）val 第 150～250 輪平均 **0.8134**，照原作補細節的（`--rcn --dwp`）**0.8107**。
> 先確定這個差距是不是真的，再找是哪個開關造成的。**程式都不用改**，三顆只差開關：
>
> | 順序 | 實驗名 | 開關 | 用來回答 |
> |---|---|---|---|
> | 第 1 顆 | `mix_cascade_pyramid_rep2` | 都不開（**跟舊的 mix_cascade_pyramid 一字不差**，只是隨機起點不同）| 舊的 0.8134 是不是那一次運氣好 |
> | 第 2 顆 | `mix_cascade_pyramid_rcn` | 只開 `--rcn` | 是不是 RCN 那個開關造成的 |
> | 第 3 顆 | `mix_cascade_pyramid_dwp` | 只開 `--dwp` | 是不是 DWP 那個開關造成的 |
>
> 🔴 **第 1 顆跑完先看結果再決定要不要跑第 2、3 顆**（見 1-4）。
> ⚠️ 代號跟 `指令_reproduce.md` 的 A、B、B2～B4 無關（11-06 從 A／B／C 改成第 1／2／3 顆，免得撞名）。
>
> 跑的機器：**AI**。🔴 **一次只跑一顆**，接在 B3（`orig_rcn_vxm1`）後面、B4 前面。每顆約 28～30 小時（舊的每步 4.0 秒）。
> 指令全部是 **cmd（命令提示字元）** 語法。最後更新：2026-11-05（筆電小尺寸試跑：三種開關組合都正常）

---

## cmd 要注意的三件事

1. **每個指令都是一整行**：整行複製、貼上、按 Enter。看起來換行的地方是畫面自動折行，不是真的斷開
2. **換磁碟要用 `cd /d`**：只打 `cd D:\...` 的話，如果目前在 C 槽，會停在原地不動
3. 虛擬環境啟動成功的話，提示字元前面會多一個 `(vxm_env)`

---

## 0. 開始前（每一顆開始前都做）

```bat
cd /d D:\chengyun\TRY_voxelmorph
```

```bat
vxm_env\Scripts\activate.bat
```

```bat
git pull
```

確認顯卡空出來了（上一顆跑完、沒有其他 python 在用 GPU）：

```bat
nvidia-smi
```

---

## 第 1 顆：mix_cascade_pyramid_rep2（舊設定再練一次）

**1-1 起跑前檢查**（不會真的開始訓練）：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_rep2 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --gpu 0 --check-only
```

`架構` 那一行要是「**串 2 顆由粗到細的網路（兩個疊在一起），共 816254 個參數**」，**不能**出現「照 RCN 原作」或「照 DWP 原作」。

**1-2 訓練**：先打

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.9
```

然後在**同一個視窗**：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_rep2 --image-lㄔㄛoss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --gpu 0
```

- log 每一行**不會**有 `ms:`（沒開 `--dwp`），正常
- 每步約 4 秒；**超過 10 秒**不對勁，停掉跟我說
- 噴 `CUDA out of memory` → 改用省顯存模式（數字一樣，慢約 1.24 倍）：先 `set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85`，再把上面那行訓練指令後面加 ` --grad-checkpoint --resume --yes`

**1-3 用 val 掃每一顆**（🔴 不要拿 test 挑）：

```bat
python ASD\test_dice.py --model-dir models\mix_cascade_pyramid_rep2 --test-dir data\mixed_preprocessed_v2\val --exp-name mix_cascade_pyramid_rep2 --step 5 --gpu 0
```

**1-4 算第 150～250 輪的 val 平均，決定要不要跑第 2、3 顆**：

```bat
python -c "import csv; r=[float(x['dice_mean']) for x in csv.DictReader(open(r'models\mix_cascade_pyramid_rep2\dice_curve_val.csv')) if int(x['epoch'])>=150]; print('points', len(r), 'mean', round(sum(r)/len(r), 4))"
```

應該印出 `points 21 mean 0.8xxx`。對照：舊的 0.8134、開兩個開關的 0.8107。

| 印出來的平均 | 代表 | 下一步 |
|---|---|---|
| **0.8120 以下** | 舊的 0.8134 主要是運氣，合併版其實沒變低 | **第 2、3 顆不用跑**，直接接 `指令_reproduce.md` 的 B4；結果帶回來 |
| **0.8120 以上** | 開關真的讓合併版變差 | 接著跑第 2、3 顆 |

拿不定主意（例如剛好 0.8118～0.8122）就先把 `dice_curve_val.csv` 帶回來給我看。

**1-5 test（只跑一次）**，`NNNN` 換成 `dice_curve_val.csv` 裡 `dice_mean` 最大的 epoch：

```bat
python ASD\test_dice.py --model models\mix_cascade_pyramid_rep2\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_cascade_pyramid_rep2 --gpu 0
```

---

## 第 2 顆：mix_cascade_pyramid_rcn（只開 --rcn）

1-4 判斷要跑才跑。先做第 0 步。

**2-1 起跑前檢查**：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_rcn --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --rcn --gpu 0 --check-only
```

`架構` 那一行要有「**照 RCN 原作：每顆輸出乘 1/2、串接的 warp 取邊界值**」，**不能**有「照 DWP 原作」。

**2-2 訓練**：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.9
```

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_rcn --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --rcn --gpu 0
```

- log **不會**有 `ms:`，正常；每步約 4 秒，超過 10 秒停掉跟我說

**2-3 val 掃每一顆**：

```bat
python ASD\test_dice.py --model-dir models\mix_cascade_pyramid_rcn --test-dir data\mixed_preprocessed_v2\val --exp-name mix_cascade_pyramid_rcn --step 5 --gpu 0
```

**2-4 test（只跑一次）**：

```bat
python ASD\test_dice.py --model models\mix_cascade_pyramid_rcn\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_cascade_pyramid_rcn --gpu 0
```

---

## 第 3 顆：mix_cascade_pyramid_dwp（只開 --dwp）

1-4 判斷要跑才跑。先做第 0 步。

**3-1 起跑前檢查**：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_dwp --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --dwp --gpu 0 --check-only
```

`架構` 那一行要有「**照 DWP 原作：合成時 δ detach、子層 1/16、1/8、1/4 也算影像項＋平滑項**」，**不能**有「照 RCN 原作」。

**3-2 訓練**：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.9
```

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_dwp --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --dwp --gpu 0
```

- log 每一行最後**要有** `ms: ...`（粗的層的 loss）；沒有的話停掉跟我說
- 每步約 4～4.5 秒，超過 10 秒停掉跟我說

**3-3 val 掃每一顆**：

```bat
python ASD\test_dice.py --model-dir models\mix_cascade_pyramid_dwp --test-dir data\mixed_preprocessed_v2\val --exp-name mix_cascade_pyramid_dwp --step 5 --gpu 0
```

**3-4 test（只跑一次）**：

```bat
python ASD\test_dice.py --model models\mix_cascade_pyramid_dwp\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_cascade_pyramid_dwp --gpu 0
```

---

## 中斷了怎麼辦（三顆都一樣）

新視窗先照第 0 步 `cd`、啟動虛擬環境，再打一次那一顆的 `set` 那行，然後把**同一行訓練指令**後面加 ` --resume`
（用過省顯存模式的話 `--grad-checkpoint` 也要留著）。`--rcn`、`--dwp` **一定要跟原本一樣**，不一樣程式會擋下來（`AssertionError: … 不能混著接`）。

⚠️ 接著練時 Adam 的狀態沒存在 .pt 裡，前幾步 loss 可能稍微跳一下，屬正常。

---

## 收尾、帶回來（每一顆跑完都一樣）

🔴 **這次 .pt 先不要刪**（之後可能要拆兩級、看每一層，挑哪幾輪還不確定）；硬碟不夠再跟我說。每一顆要帶回主機的東西（`<實驗名>` 換成 `mix_cascade_pyramid_rep2` 等）：

```
models\<實驗名>\NNNN.pt               最佳 epoch 的權重
models\<實驗名>\dice_curve_val.csv    val 曲線（判斷用的是第 150～250 輪平均）
models\<實驗名>\dice_NNNN.csv         test 結果
log\<實驗名>.txt                      訓練 stdout（要比訓練 loss）
log\<實驗名>_script.txt               當初下的指令
```

帶回來之後，我在筆電上跟舊的 mix_cascade_pyramid、mix_cascade_pyramid_rcn_dwp 比：val 第 150～250 輪平均、訓練影像項、拆兩級。

---

## 注意事項

- 🔴 **實驗名稱一定要用新的**（`mix_cascade_pyramid_rep2`、`_rcn`、`_dwp`），不要用舊的名字，會蓋掉舊的 log 和權重
- 🔴 **開關不能多也不能少**：A 都不開、B 只有 `--rcn`、C 只有 `--dwp`；起跑前檢查那一行一定要看
- 其他參數跟舊的一字不差：`--image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid`
- **不要跟其他訓練同時跑**（同一張卡）
