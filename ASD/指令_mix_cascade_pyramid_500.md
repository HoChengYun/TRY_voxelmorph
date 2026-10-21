# mix_cascade_pyramid_500 操作單（cmd 版）：mix_cascade_pyramid 接著訓練到 500 輪

> **這顆在問什麼**：mix_pyramid 接著練到 500 輪（mix_pyramid_500）之後，test 0.8185，跟只練 250 輪的 mix_cascade_pyramid（0.8192）打平。
> 但 mix_cascade_pyramid 的 val 在 200～250 輪還在上升（區間平均 0.8127 → 0.8144），兩顆練的長度不一樣，比得不公平。
> 讓 mix_cascade_pyramid 也練到 500 輪，看「串兩顆＋由粗到細」在同樣長度下是不是還比較好（10/28 簡報第 6 頁留的問題）。
>
> **做法**：從 mix_cascade_pyramid 的**第 250 輪**權重接著練，其他設定完全不變（串 2 顆、每顆由粗到細、速度場、全尺寸積分、λ 1、同一份資料）。
> 用**新的實驗名稱 `mix_cascade_pyramid_500`**：包裝腳本續跑時會重寫 `log\<實驗名>.txt`，用原本的名字會把原本 250 輪的 log 蓋掉。
>
> 跑的機器：**AI**。🔴 **等 TransMorph 跑完再開始**（TransMorph 約 13 GB、這顆約 19 GB，24 GB 的卡放不下兩顆）。
> 指令全部是 **cmd（命令提示字元）** 語法。
> 最後更新：2026-10-21（筆電實測：從 0140.pt 接著練、包裝腳本 `--resume --check-only` 正常）

---

## cmd 要注意的三件事

1. **每個指令都是一整行**：整行複製、貼上、按 Enter。看起來換行的地方是畫面自動折行，不是真的斷開
2. **換磁碟要用 `cd /d`**：只打 `cd D:\...` 的話，如果目前在 C 槽，會停在原地不動
3. 虛擬環境啟動成功的話，提示字元前面會多一個 `(vxm_env)`

---

## 0. 開始前

```bat
cd /d D:\chengyun\TRY_voxelmorph
```

```bat
vxm_env\Scripts\activate.bat
```

```bat
git pull
```

確認顯卡空出來了（TransMorph 跑完、沒有其他 python 在用 GPU）：

```bat
nvidia-smi
```

---

## 1. 準備起點

先看原本的資料夾還有哪些權重：

```bat
dir models\mix_cascade_pyramid\*.pt
```

🔴 **要有 `0250.pt`**（訓練的最後一輪）。只剩 `0140.pt` 的話（原本操作單收尾時把其他刪了）**先停下來跟我說**，
從 0140 接著練等於 140～250 輪重練一次，要另外決定。

建新資料夾、複製起點：

```bat
mkdir models\mix_cascade_pyramid_500
```

```bat
copy models\mix_cascade_pyramid\0250.pt models\mix_cascade_pyramid_500\
```

⚠️ `models\mix_cascade_pyramid_500\` 裡**只能有一顆** .pt（續跑會從裡面編號最大的那顆開始）。

---

## 2. 起跑前檢查（不會真的開始訓練）

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_500 --image-loss ncc --lambda 1.0 --epochs 500 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --gpu 0 --resume --check-only
```

要看到這兩行：
- `[i] --resume：從 epoch 250 續跑`
- `epochs      : 500 (從 250 開始，還要跑 250)`

---

## 3. 訓練

先打這一行（跟原本 mix_cascade_pyramid 一樣用 **0.9**，只管記憶體、不改計算）：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.9
```

然後在**同一個視窗**開始訓練：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_500 --image-loss ncc --lambda 1.0 --epochs 500 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --gpu 0 --resume
```

包裝腳本會自己存 `log\mix_cascade_pyramid_500.txt`、`log\mix_cascade_pyramid_500_script.txt`、`models\mix_cascade_pyramid_500\`。

**要跑多久**：250 輪，約 **28 小時**（原本那次每步約 4.0 秒、250 輪 27.7 小時）。包裝腳本印的「預估 42.9 小時」是舊的估計，不用理它。

| 看到 | 怎麼辦 |
|---|---|
| 噴 `CUDA out of memory` | 確認 TransMorph 真的停了；還是不行就用下面的省顯存模式 |
| 每步**超過 8 秒** | 不對勁：`Ctrl + C` 停掉，`echo %PYTORCH_CUDA_ALLOC_CONF%` 檢查，先跟我說，**不要自己改參數** |

**省顯存模式**（OOM 才用）：算出來的數字一樣，只是慢約 1.24 倍。先打 `set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85`，再：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_cascade_pyramid_500 --image-loss ncc --lambda 1.0 --epochs 500 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid --grad-checkpoint --gpu 0 --resume --yes
```

**中斷了怎麼辦**：新視窗先照第 0 步 `cd`、啟動虛擬環境，再打一次 `set` 那行，然後跑一次第 3 步的訓練指令（本來就有 `--resume`；用過省顯存模式的話 `--grad-checkpoint` 也要留著）。

⚠️ 接著練時，優化器（Adam）的內部狀態沒有存在 .pt 裡，會重新開始累積。前幾步 loss 可能稍微跳一下，屬正常
（筆電實測從 0140.pt 接著練：第一步 loss −0.2533，原本第 140 輪前 3 步是 −0.2549、−0.2463、−0.2518，同一個水準）。

---

## 4. 訓練完 → 用驗證集挑 epoch

🔴 **不要拿 test 挑 epoch。**

```bat
python ASD\test_dice.py --model-dir models\mix_cascade_pyramid_500 --test-dir data\mixed_preprocessed_v2\val --exp-name mix_cascade_pyramid_500 --step 5 --gpu 0
```

輸出 `models\mix_cascade_pyramid_500\dice_curve_val.csv`。取 `dice_mean` 最大的那個 epoch。

---

## 5. test 評估（只跑一次）

這顆是為了跟 mix_pyramid_500 公平比較，**不管 val 有沒有比較高，test 都跑一次**。把 `NNNN` 換成第 4 步挑出來的 epoch：

```bat
python ASD\test_dice.py --model models\mix_cascade_pyramid_500\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_cascade_pyramid_500 --gpu 0
```

---

## 6. 收尾、帶回來

挑好 epoch 之後，只留最佳那顆 `.pt`，其他刪掉（每個 3.1 MB）。

要帶回主機的東西：

```
models\mix_cascade_pyramid_500\NNNN.pt               最佳 epoch 的權重
models\mix_cascade_pyramid_500\dice_curve_val.csv    挑 epoch 的依據（250～500 輪）
models\mix_cascade_pyramid_500\dice_NNNN.csv         test 結果
log\mix_cascade_pyramid_500.txt                      訓練 stdout
log\mix_cascade_pyramid_500_script.txt               當初下的指令
```

帶回來之後，跟原本 0～250 輪的曲線接起來、跟 mix_pyramid_500 逐人配對比較，我來算。

---

## 注意事項

- 🔴 **`--exp-name mix_cascade_pyramid_500`**：不要用 `mix_cascade_pyramid`，會蓋掉原本的 log
- 🔴 **`--epochs 500 --resume`**：兩個都要；`--epochs` 是「練到第幾輪」，不是「再練幾輪」
- 🔴 其他參數跟原本 mix_cascade_pyramid 一字不差：`--image-loss ncc --lambda 1.0 --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2 --stage pyramid`
- **不要跟其他訓練同時跑**（同一張卡）
