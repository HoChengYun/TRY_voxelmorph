# 照原作程式重跑三顆（cmd 版）：mix_pyramid_dwp → mix_cascade_rcn → mix_cascade_pyramid_rcn_dwp

> **為什麼要重跑**：mix_cascade／mix_pyramid／mix_cascade_pyramid 的程式是照論文文字寫的，原作程式是事後才對照（手冊 §25.5）。
> 2026-10-21 重新讀了原作程式（RCN、DWP），補上三個細節，用**新的實驗名稱**重跑，舊的三顆保留、一起報告（手冊 §29）：
>
> | 新實驗名 | 對應的舊實驗 | 補上的細節（出處） |
> |---|---|---|
> | `mix_pyramid_dwp` | mix_pyramid | 合成時 δ detach；粗的層（1/16、1/8、1/4）也算影像項＋平滑項，權重 1/16、1/8、1/4（DWP：Jian et al., WBIR 2024，github.com/BailiangJ/rethink-reg） |
> | `mix_cascade_rcn` | mix_cascade | 每顆輸出乘 1/2；串接的 warp 取邊界值（RCN：Zhao et al., ICCV 2019，github.com/microsoft/Recursive-Cascaded-Networks） |
> | `mix_cascade_pyramid_rcn_dwp` | mix_cascade_pyramid | 上面兩組都加（兩顆之間照 RCN，每一顆內部照 DWP；粗層 loss 只算最後一顆） |
>
> 其他設定跟舊的完全一樣（速度場、全尺寸積分、λ 1、同一份資料、250 輪、val 挑 epoch）。程式的開關是 `--dwp`、`--rcn`，不加就是舊版本。
>
> 跑的機器：**AI**。🔴 **等 TransMorph 跑完再開始**，三顆**一次跑一顆**，依序跑。合計約 2.5～3 天。
> ⚠️ `指令_mix_cascade_pyramid_500.md` **先不要跑**（那是舊版本的程式，改跑這三顆）。
> 指令全部是 **cmd（命令提示字元）** 語法。最後更新：2026-10-21（筆電驗證：舊模型結果逐 voxel 不變、三種組合小尺寸訓練正常、接續訓練正常）

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

確認顯卡空出來了（TransMorph 或上一顆跑完、沒有其他 python 在用 GPU）：

```bat
nvidia-smi
```

---

## A. mix_pyramid_dwp（約 14 小時）

**A-1 起跑前檢查**（不會真的開始訓練）：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_pyramid_dwp --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch pyramid --dwp --gpu 0 --check-only
```

`架構` 那一行要看到「**照 DWP 原作：合成時 δ detach、子層 1/16、1/8、1/4 也算影像項＋平滑項**」。

**A-2 訓練**：先打

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85
```

然後在**同一個視窗**：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_pyramid_dwp --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch pyramid --dwp --gpu 0
```

- 每一行 log 最後要有 `ms: ...`（粗的層的 loss）。沒有的話 `Ctrl + C` 停掉跟我說
- 每步約 2 秒（原本 mix_pyramid 2.0 秒）；**超過 6 秒**不對勁，停掉跟我說

**A-3 用 val 挑 epoch**（🔴 不要拿 test 挑）：

```bat
python ASD\test_dice.py --model-dir models\mix_pyramid_dwp --test-dir data\mixed_preprocessed_v2\val --exp-name mix_pyramid_dwp --step 5 --gpu 0
```

取 `models\mix_pyramid_dwp\dice_curve_val.csv` 裡 `dice_mean` 最大的 epoch。

**A-4 test（只跑一次）**，`NNNN` 換成 A-3 挑出來的 epoch：

```bat
python ASD\test_dice.py --model models\mix_pyramid_dwp\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_pyramid_dwp --gpu 0
```

---

## B. mix_cascade_rcn（約 18～21 小時）

先做第 0 步。

**B-1 起跑前檢查**：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_rcn --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --rcn --gpu 0 --check-only
```

`架構` 那一行要看到「**照 RCN 原作：每顆輸出乘 1/2、串接的 warp 取邊界值**」。

**B-2 訓練**：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85
```

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_rcn --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --rcn --gpu 0
```

- 這顆的 log 沒有 `ms:`（cascade 沒有粗的層），正常
- 每步**超過 8 秒**不對勁，停掉跟我說

**B-3 val 挑 epoch**：

```bat
python ASD\test_dice.py --model-dir models\mix_cascade_rcn --test-dir data\mixed_preprocessed_v2\val --exp-name mix_cascade_rcn --step 5 --gpu 0
```

**B-4 test（只跑一次）**：

```bat
python ASD\test_dice.py --model models\mix_cascade_rcn\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_cascade_rcn --gpu 0
```

---

## C. mix_cascade_pyramid_rcn_dwp（約 30 小時）

先做第 0 步。

**C-1 起跑前檢查**：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_rcn_dwp --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --rcn --dwp --gpu 0 --check-only
```

`架構` 那一行要**兩個都看到**：「照 RCN 原作：每顆輸出乘 1/2、串接的 warp 取邊界值」和「照 DWP 原作：合成時 δ detach、子層 1/16、1/8、1/4 也算影像項＋平滑項」（開始訓練後第一行還會多寫「只算最後一顆」）。

**C-2 訓練**（這顆比較大，用 **0.9**）：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.9
```

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_rcn_dwp --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --rcn --dwp --gpu 0
```

- log 最後要有 `ms: ...`；每步約 4～4.5 秒（原本 4.0 秒）；**超過 10 秒**停掉跟我說
- 噴 `CUDA out of memory` → 改用省顯存模式（數字一樣，慢約 1.24 倍）：先 `set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85`，再

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_rcn_dwp --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --rcn --dwp --grad-checkpoint --gpu 0 --resume --yes
```

**C-3 val 挑 epoch**：

```bat
python ASD\test_dice.py --model-dir models\mix_cascade_pyramid_rcn_dwp --test-dir data\mixed_preprocessed_v2\val --exp-name mix_cascade_pyramid_rcn_dwp --step 5 --gpu 0
```

**C-4 test（只跑一次）**：

```bat
python ASD\test_dice.py --model models\mix_cascade_pyramid_rcn_dwp\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_cascade_pyramid_rcn_dwp --gpu 0
```

---

## 中斷了怎麼辦（三顆都一樣）

新視窗先照第 0 步 `cd`、啟動虛擬環境，再打一次那一顆的 `set` 那行，然後把**同一行訓練指令**後面加 ` --resume`
（用過省顯存模式的話 `--grad-checkpoint` 也要留著）。`--rcn`、`--dwp` **一定要跟原本一樣**，少一個程式會擋下來（`AssertionError: … 不能混著接`）。

⚠️ 接著練時 Adam 的狀態沒存在 .pt 裡，前幾步 loss 可能稍微跳一下，屬正常。

---

## 收尾、帶回來（每一顆跑完都一樣）

挑好 epoch 之後，只留最佳那顆 `.pt`，其他刪掉。每一顆要帶回主機的東西（`<實驗名>` 換成 `mix_pyramid_dwp` 等）：

```
models\<實驗名>\NNNN.pt               最佳 epoch 的權重
models\<實驗名>\dice_curve_val.csv    挑 epoch 的依據
models\<實驗名>\dice_NNNN.csv         test 結果
log\<實驗名>.txt                      訓練 stdout
log\<實驗名>_script.txt               當初下的指令
```

帶回來之後，HD95、SDlogJ、沒參與訓練的結構、視覺化、loss 圖、Dice 曲線我在筆電上補，跟舊的三顆逐人配對比較。

---

## 注意事項

- 🔴 **實驗名稱一定要用新的**（`mix_pyramid_dwp`、`mix_cascade_rcn`、`mix_cascade_pyramid_rcn_dwp`），不要用舊的名字，會蓋掉舊的 log 和權重
- 🔴 **`--dwp`／`--rcn` 不能漏**：漏了就是舊版本，等於白跑
- 其他參數跟舊的一字不差：`--image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1`
- **不要跟其他訓練同時跑**（同一張卡）
