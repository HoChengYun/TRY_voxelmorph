# mix_wide_vel 操作單（cmd 版）

> **這顆在問什麼**：老師 09-30 的紅字（p25 加寬）「改看看速度」——**加寬 2 倍的 U-Net 改成速度場版**。
>
> 四顆都是全尺寸積分、平滑權重 1，剛好湊成 2 × 2：
>
> | | 預設寬度 | 加寬 2 倍 |
> |---|---|---|
> | **位移場** | mix_exp3：0.8062、擠爆 0.199% | mix_wide：0.8114、擠爆 0.187% |
> | **速度場** | mix_exp6：（跑完補）| **mix_wide_vel：？** |
>
> - **mix_wide_vel − mix_wide ＝ 只差版本**（都是加寬 2 倍）
> - **mix_wide_vel − mix_exp6 ＝ 只差寬度**（都是速度場）
>
> 平滑權重用 1，是為了跟 mix_wide、mix_exp6 都只差一件事，**不用等 mix_exp7 的結果**。
>
> 跑的機器：**AI**，**等 mix_exp7 跑完再開始**（同一張顯卡）。指令全部是 **cmd（命令提示字元）** 語法。
> 最後更新：2026-10-01

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

確認 mix_exp7 已經跑完（`log\mix_exp7.txt` 最後一行是 `epoch: 0250  step: 100/100`），顯卡空出來了再開始。

🔴 **不要跟 mix_exp6／mix_exp7 同時跑**：8.7 GB ＋ 14.1 GB，24 GB 的卡幾乎塞滿，兩顆都會變慢。

需要 `data\mixed_preprocessed_v2\`（train 418 / val 51 / test 51），跟 mix_wide、mix_exp6 同一份，AI 上已經有了。

---

## 1. 起跑前檢查（不會真的開始訓練）

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_wide_vel --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --enc 32 64 64 64 --dec 64 64 64 64 64 32 32 --gpu 0 --check-only
```

確認它印出來的 atlas 是 `atlas_mni152_09c_v3.npz`。

**顯存**：2026-10-01 在筆電上用小尺寸量、外插到全尺寸：

| 設定 | 顯存 |
|---|---|
| mix_exp3（位移場、預設寬度）| 7.3 GB |
| mix_exp5／6／7（速度場、預設寬度）| 8.7 GB |
| mix_wide（位移場、加寬 2 倍）| 13.5 GB |
| **mix_wide_vel（速度場、加寬 2 倍）** | **14.1 GB** |

AI 是 24 GB，放得下。⚠️ `--check-only` 只會印出卡有多大，不會真的建模型試跑，警告門檻 7.5 GB 是給預設寬度的，**看到警告不用理**。

🔴 **`--int-downsize` 一定要寫 1。** 預設是 2，漏寫的話速度場會縮小一半再積分、平滑權重也會變 2，兩件事一起變。

🔴 **`--enc`、`--dec` 要跟 mix_wide 一模一樣**：`--enc 32 64 64 64 --dec 64 64 64 64 64 32 32`。

---

## 2. 訓練

🔴 **先打這一行**（限制 PyTorch 最多只佔顯卡的 85%，原因見這一節最後）：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85
```

- 只對**這個 cmd 視窗**有效，關掉就沒了。**換新視窗（例如中斷後要續跑）要再打一次**
- 等號前後**不要有空白**
- 想確認有沒有打對：`echo %PYTORCH_CUDA_ALLOC_CONF%`，應該印出 `per_process_memory_fraction:0.85`

然後在**同一個視窗**開始訓練：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_wide_vel --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --enc 32 64 64 64 --dec 64 64 64 64 64 32 32 --gpu 0
```

包裝腳本會自動處理這些，**不用自己加 `>` 導向**：

| 它會做的事 | 位置 |
|---|---|
| 存 stdout | `log\mix_wide_vel.txt` |
| 存當初下的指令 | `log\mix_wide_vel_script.txt` |
| 存權重 | `models\mix_wide_vel\` |
| 帶 atlas | 預設就是 `IXI\atlas_mni152_09c_v3.npz` |

**要跑多久**（看第一個 epoch 印出來的 `time:`，第 1 步常常特別快，看第 2 步以後）：

| | 每步 | 250 epoch |
|---|---|---|
| **有打 `set` 那行（預期）** | **約 2.7 秒** | **約 19 小時** |
| 沒打、或沒生效 | 約 4 秒 | 約 28 小時 |

2.7 秒 ＝ 筆電量的倍數 1.72 × mix_exp3 在 AI 上的 1.57 秒。

| 看到 | 怎麼辦 |
|---|---|
| 每步 **4 秒左右** | `set` 那行沒生效：`Ctrl + C` 停掉，用 `echo` 檢查，重打 `set`，再跑下面的續跑指令 |
| 噴 `CUDA out of memory` | 85% 不夠用：把 `set` 那行的 `0.85` 改成 `0.9` 重打，再跑續跑指令 |
| 每步**超過 8 秒** | 不對勁，先停下來跟我說，**不要自己改參數** |

**中斷了怎麼辦**：新視窗先照第 0 步 `cd`、啟動虛擬環境，**再打一次 `set` 那行**，然後用下面這行，
會從最後一個 `.pt` 接著跑（還沒存到任何 `.pt` 的話，它會自己從頭開始）：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_wide_vel --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --enc 32 64 64 64 --dec 64 64 64 64 64 32 32 --gpu 0 --resume
```

**為什麼要多打 `set` 那行**（2026-10-01 在筆電上查到的）：

- PyTorch 訓練時會先**多佔一些顯存放著備用**，佔的量大約是真正用到的 **2.4 倍**
- 這顆真正要用 14.1 GB，不限制的話會想佔到約 33 GB；**超過 24 GB 的部分，Windows 會拿一般記憶體來頂**，
  一般記憶體慢很多，整顆就變慢。mix_wide 預估每步 2.5 秒、實際 3.75 秒，很可能就是這個
- `set` 那行讓 PyTorch 最多只佔 85%（約 20 GB），不夠時先退掉自己囤著沒用的位子
- **只改記憶體怎麼管，算出來的數字不變**，結果照樣能跟 mix_wide 比

筆電實測（同一顆模型、把尺寸縮小，讓筆電的 8 GB 也出現同樣的溢位）：

| | 每步 |
|---|---|
| 不加 | 13.8 秒（溢位）|
| **加了** | **0.93 秒**（跟沒溢位一樣快）|
| 不會溢位的尺寸也加 | 0.40 → 0.39 秒，沒有壞處 |

---

## 3. 訓練完 → 用驗證集挑 epoch

🔴 **不要拿 test 挑 epoch。** test 只能跑一次。

```bat
python ASD\test_dice.py --model-dir models\mix_wide_vel --test-dir data\mixed_preprocessed_v2\val --exp-name mix_wide_vel --step 5 --gpu 0
```

輸出 `models\mix_wide_vel\dice_curve_val.csv`。接著把起點複製過來、畫圖（四行可以一起貼）：

```bat
copy models\mix_exp2\dice_baseline.csv models\mix_wide_vel\
copy models\mix_exp2\dice_baseline_val.csv models\mix_wide_vel\
python ASD\plot_dice_curve.py --model-dir models\mix_wide_vel --label "加寬 速度場"
python ASD\plot_loss_curve.py --logs log\mix_wide_vel.txt log\mix_wide.txt log\mix_exp6.txt --labels 加寬速度場 加寬位移場 預設寬速度場 --out models\mix_wide_vel\loss_widevel_wide_exp6.png
```

- 起點跟模型無關，mix_exp2 那兩份直接用。`plot_dice_curve.py` 會自己讀：test 起點 0.6882、驗證集起點 0.6817
- 🔴 **不要加 `--baseline 0.6817`**：那個參數是給 test 的，會把驗證集的起點標成 test
- 🔴 最後一行**一定要 `--out`**，同時畫多份 log 沒給會直接停
- ⚠️ `log\mix_wide.txt`、`log\mix_exp6.txt` 要在 AI 上。沒有的話最後一行先跳過，帶回來我這邊畫

**挑法**：取 `dice_mean` 最大的那個 epoch。相鄰 epoch 的抖動約 ±0.003，曲線平的時候別太在意差 0.001 的名次。

---

## 4. test 評估（只跑一次）

把 `NNNN` 換成第 3 步挑出來的 epoch（例如 `0225`）：

```bat
python ASD\test_dice.py --model models\mix_wide_vel\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_wide_vel --gpu 0
```

輸出 `models\mix_wide_vel\dice_NNNN.csv`（逐人、逐結構）。

---

## 5. 要比的數字、怎麼判讀

| | mix_wide | **mix_wide_vel** | mix_exp6 |
|---|---|---|---|
| 版本 | 位移場 | 速度場 | 速度場 |
| 寬度 | 加寬 2 倍 | 加寬 2 倍 | 預設 |
| test Dice | 0.8114 | **？** | （跑完補）|
| 擠爆比例 | 0.187% | **？** | （跑完補）|

👉 **跟 mix_wide 比（只差版本）**：
- **差不多或更高、而且擠爆接近 0%** → 加寬＋速度場是目前最好的組合：一樣準，又不擠爆
- **明顯比較低** → 加寬的好處在速度場上縮水，速度場的平滑限制比較緊

👉 **跟 mix_exp6 比（只差寬度）**：位移場那邊加寬多了 +0.0053，看速度場這邊加寬能多多少。

帶回來之後逐人配對（51 位）我來算。

---

## 6. 視覺化（挑好 epoch 之後）

跟之前幾顆同樣四位。兩行都要把 `NNNN` 換掉，可以一起貼：

```bat
for %s in (T054 D031 VNT045 sub-0043) do python ASD\visualize_dice.py --model models\mix_wide_vel\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_wide_vel\vis_%s --gpu 0
for %s in (T054 D031 VNT045 sub-0043) do python draw-img\visualize_reg_ixi.py --model models\mix_wide_vel\NNNN.pt --atlas IXI\atlas_mni152_09c_v3.npz --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_wide_vel\vis_%s --gpu 0
```

每位 8 張（第一行 3 張、第二行 5 張）。
- `%s` 是 cmd 的迴圈變數，直接貼上就好。⚠️ 如果改存成 `.bat` 檔再執行，要改成 `%%s`
- 這步也可以不做，把 `.pt` 帶回來我這邊跑

---

## 7. 收尾

挑好 epoch、test 跑完之後，只留最佳那顆 `.pt`，其他刪掉（250 個檔案很佔空間）。

要帶回主機的東西：

```
models\mix_wide_vel\NNNN.pt                最佳 epoch 的權重
models\mix_wide_vel\dice_curve_val.csv     挑 epoch 的依據
models\mix_wide_vel\dice_NNNN.csv          test 結果
models\mix_wide_vel\*.png                  曲線圖
models\mix_wide_vel\vis_*\                 視覺化（有做的話）
log\mix_wide_vel.txt                       訓練 stdout
log\mix_wide_vel_script.txt                當初下的指令
```

---

## 注意事項

- 🔴 **訓練前先打 `set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85`**（第 2 步），換視窗要重打。
  它只管記憶體、不改計算，所以不算多一個變因
- 🔴 **`--int-steps 7 --int-downsize 1`**：這顆跟 mix_wide 唯一的差別是 `--int-steps`（0 → 7）
- 🔴 **`--enc 32 64 64 64 --dec 64 64 64 64 64 32 32`**：跟 mix_wide 一樣
- 🔴 **`--lambda 1.0`**：實際平滑權重 = 1.0 × 1 = 1，跟 mix_wide、mix_exp6 一樣
- 🔴 **`--image-loss ncc`** 要明寫（預設是 mse）
- **`--atlas-seg` 不用換**：這是 FreeSurfer 那組，用預設的 `atlas_mni152_09c_v3_seg.npz`
- 其他全部跟 mix_wide 一樣：同一份資料、250 epoch、學習率 1e-4
- 📌 跑完把結果補進 `ASD\ASD相關手冊.md` §23 和 `models\README.md`
