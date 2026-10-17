# mix_pyramid_500 操作單（cmd 版）：mix_pyramid 接著訓練到 500 輪

> **這顆在問什麼**：mix_pyramid（由粗到細）的 val 最佳出現在第 245 輪（0.8128），訓練只到 250 輪，看起來還在進步；
> mix_cascade 則在 170 輪就持平。接著練到 500 輪，看 pyramid 還有沒有進步空間——
> 「pyramid 跟 cascade 打平」是不是只因為 pyramid 練得不夠久（手冊 §28.1）。
>
> **做法**：從 mix_pyramid 的最後一顆權重接著練，其他設定完全不變（速度場、全尺寸積分、λ 1、同一份資料）。
> 用**新的實驗名稱 `mix_pyramid_500`**：包裝腳本續跑時會重寫 `log\<實驗名>.txt`，用原本的名字會把原本 250 輪的 log 蓋掉。
>
> 跑的機器：**AI**。🔴 **等 mix_cascade_pyramid 跑完再開始**（它用 19 GB，這顆約 10 GB，24 GB 的卡放不下兩顆）。
> 指令全部是 **cmd（命令提示字元）** 語法。
> 最後更新：2026-10-16

---

## cmd 要注意的三件事

1. **每個指令都是一整行**：整行複製、貼上、按 Enter。看起來換行的地方是畫面自動折行，不是真的斷開
2. **換磁碟要用 `cd /d`**：只打 `cd D:\...` 的話，如果目前在 C 槽，會停在原地不動
3. 虛擬環境啟動成功的話，提示字元前面會多一個 `(vxm_env)`

---

## 0. 開始前

```bat
cd /d D:\chengyun\TRY_voxelmorph
vxm_env\Scripts\activate.bat
git pull
```

確認顯卡空出來了（mix_cascade_pyramid 跑完、沒有其他 python 在用 GPU）：

```bat
nvidia-smi
```

---

## 1. 準備起點

先看原本的資料夾還有哪些權重：

```bat
dir models\mix_pyramid\*.pt
```

建新資料夾：

```bat
mkdir models\mix_pyramid_500
```

**有 `0250.pt` 的話**（優先用這顆，接在訓練的最後一輪）：

```bat
copy models\mix_pyramid\0250.pt models\mix_pyramid_500\
```

**沒有 `0250.pt`、只有 `0245.pt` 的話**（原本操作單第 7 步把其他刪了）：

```bat
copy models\mix_pyramid\0245.pt models\mix_pyramid_500\
```

⚠️ `models\mix_pyramid_500\` 裡**只能有一顆** .pt（續跑會從裡面編號最大的那顆開始）。

---

## 2. 起跑前檢查（不會真的開始訓練）

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_pyramid_500 --image-loss ncc --lambda 1.0 --epochs 500 --int-steps 7 --int-downsize 1 --arch pyramid --gpu 0 --resume --check-only
```

要看到這兩行：
- `[i] --resume：從 epoch 250 續跑`（從 0245 開始的話是 245）
- `epochs      : 500 (從 250 開始，還要跑 250)`

---

## 3. 訓練

先打這一行（跟之前一樣，只管記憶體、不改計算）：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85
```

然後在**同一個視窗**開始訓練：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_pyramid_500 --image-loss ncc --lambda 1.0 --epochs 500 --int-steps 7 --int-downsize 1 --arch pyramid --gpu 0 --resume
```

包裝腳本會自己存 `log\mix_pyramid_500.txt`、`log\mix_pyramid_500_script.txt`、`models\mix_pyramid_500\`。

**要跑多久**：250 輪，預估約 19 小時（每步約 2.7 秒）。原本那次 log 上的時間不準（當時的程式還沒有計時修正），這次 `git pull` 之後的 log 才準。

| 看到 | 怎麼辦 |
|---|---|
| 噴 `CUDA out of memory` | 確認 mix_cascade_pyramid 真的停了；把 `set` 那行的 `0.85` 改成 `0.9` 重打，再跑一次上面的指令（有 `--resume`，會從最後一顆接著跑）|
| 每步**超過 6 秒** | 不對勁：`Ctrl + C` 停掉，`echo %PYTORCH_CUDA_ALLOC_CONF%` 檢查，先跟我說，**不要自己改參數** |

**中斷了怎麼辦**：新視窗先照第 0 步 `cd`、啟動虛擬環境，再打一次 `set` 那行，然後跑一次第 3 步的訓練指令（本來就有 `--resume`）。

⚠️ 接著練時，優化器（Adam）的內部狀態沒有存在 .pt 裡，會重新開始累積。前幾步 loss 可能稍微跳一下，屬正常。

---

## 4. 訓練完 → 用驗證集挑 epoch

🔴 **不要拿 test 挑 epoch。**

```bat
python ASD\test_dice.py --model-dir models\mix_pyramid_500 --test-dir data\mixed_preprocessed_v2\val --exp-name mix_pyramid_500 --step 5 --gpu 0
```

輸出 `models\mix_pyramid_500\dice_curve_val.csv`。

**挑法**：取 `dice_mean` 最大的那個 epoch。**拿它跟原本的最佳 0.8128（第 245 輪）比：**

| val 最佳 | 代表 | 下一步 |
|---|---|---|
| 明顯高於 0.8128（例如 0.815 以上）| 多練有用 | 照第 5 步跑 test |
| 跟 0.8128 差不多或更低 | 250 輪已經差不多了 | **不用跑 test**，帶 `dice_curve_val.csv` 和 log 回來就好 |

---

## 5. test 評估（只有第 4 步判定要跑時才跑；只跑一次）

把 `NNNN` 換成第 4 步挑出來的 epoch：

```bat
python ASD\test_dice.py --model models\mix_pyramid_500\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_pyramid_500 --gpu 0
```

---

## 6. 收尾、帶回來

挑好 epoch 之後，只留最佳那顆 `.pt`，其他刪掉（每個 1.6 MB）。

要帶回主機的東西：

```
models\mix_pyramid_500\NNNN.pt               最佳 epoch 的權重
models\mix_pyramid_500\dice_curve_val.csv    挑 epoch 的依據（250～500 輪）
models\mix_pyramid_500\dice_NNNN.csv         test 結果（有跑的話）
log\mix_pyramid_500.txt                      訓練 stdout
log\mix_pyramid_500_script.txt               當初下的指令
```

帶回來之後，跟原本 0～250 輪的曲線接起來、逐人配對比較，我來算。

---

## 注意事項

- 🔴 **`--exp-name mix_pyramid_500`**：不要用 `mix_pyramid`，會蓋掉原本的 log
- 🔴 **`--epochs 500 --resume`**：兩個都要；`--epochs` 是「練到第幾輪」，不是「再練幾輪」
- 🔴 其他參數跟原本 mix_pyramid 一字不差：`--image-loss ncc --lambda 1.0 --int-steps 7 --int-downsize 1 --arch pyramid`
- **不要跟其他訓練同時跑**（同一張卡）
- 📌 這顆練得比其他模型久（500 vs 250 輪），比較時要註明；如果真的有進步，之後可能也要讓 cascade 練一樣久才公平
