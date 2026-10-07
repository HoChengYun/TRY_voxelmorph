# mix_exp8／mix_exp9 操作單（cmd 版）：訓練時也用 FreeSurfer 標籤

> **這兩顆在問什麼**：到目前為止，所有模型訓練時都**只看影像**。論文（TMI 2019 式 10）可以在訓練時多加一項
> 「**搬過去的標籤要跟 atlas 的標籤重疊**」，測試時照樣只用影像、不需要標籤。
> 論文的結果：用到的結構 Dice 顯著變好（p < 10⁻⁹），換成人工標記的另一批資料測也一樣進步。
>
> | 實驗 | 基礎設定 | 標籤權重 γ | 跟誰比 |
> |---|---|---|---|
> | **mix_exp8** | 同 mix_exp6（速度場、全尺寸、平滑權重 1）| **0.5** | 跟 mix_exp6 **只差「有沒有用標籤」** |
> | **mix_exp9** | 同上 | **5** | 跟 mix_exp8 只差標籤權重 |
>
> **為什麼 γ 是 0.5 和 5，不是論文的 0.01 和 0.1**：論文的 γ 是搭 MSE（λ = 0.02）。我們用 NCC（λ = 1），
> 損失的量級差約 50 倍，γ 要跟 λ 一起放大，「γ / λ」才跟論文一樣：論文 0.01 → 我們 0.5、論文 0.1 → 我們 5。
> 直接照抄 0.01 的話，標籤項小到幾乎沒有作用（跟 CLAUDE.md「λ 的尺度取決於 image-loss」同一件事）。
>
> 跑的機器：**AI**，**等 mix_wide_vel 跑完再開始**（同一張顯卡）。指令全部是 **cmd（命令提示字元）** 語法。
> 最後更新：2026-10-07（第 2 步加「一行串接兩顆」的指令）

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
dir ASD\train_semisup.py
```

🔴 **這兩顆要用新程式 `ASD\train_semisup.py`**，`git pull` 之後最後一行要看得到這個檔案。
看不到的話，就是筆電那邊還沒 push，先停下來跟我說。

需要 `data\mixed_preprocessed_v2\`（跟 mix_exp6 同一份，AI 上已經有了），而且 **npz 裡要有 `seg`**（FreeSurfer 組的有，tigerbx 組的不能拿來跑這個）。

---

## 1. 起跑前檢查（不會真的開始訓練）

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp8 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --seg-weight 0.5 --gpu 0 --check-only
```

要看到這三行：

```
[v] ...\ASD\train_semisup.py
[v] ...\IXI\atlas_mni152_09c_v3_seg.npz
[v] 訓練資料有 seg（A002.npz）
```

以及「標籤權重 γ : 0.5」。

**顯存**：約 **12.2 GB**（預留約 18.6 GB，在 24 GB 以內）（mix_exp6 是 8.7 GB，多出來的是 30 個結構的標籤一起搬）。

🔴 **`--int-downsize` 一定要寫 1**、**`--lambda 1.0`**：跟 mix_exp6 一樣，才能只差「有沒有用標籤」。

---

## 2. 訓練

先打這一行（限制 PyTorch 最多只佔顯卡的 85%，原因見 `ASD\指令_mix_wide_vel.md` 第 2 步；換新視窗要再打一次）：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85
```

然後在同一個視窗，**用一行把兩顆串起來**（建議，2026-10-07 加）：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp8 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --seg-weight 0.5 --gpu 0 && python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp9 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --seg-weight 5 --gpu 0
```

- `&&`：前面的 mix_exp8 **正常跑完**才會接著開始 mix_exp9，中間不會空等（兩顆接著跑約 28 小時）
- mix_exp8 中途出錯的話，mix_exp9 **不會**開始（`run_train.py` 會回傳非零結束碼），先停下來跟我說
- 兩顆共用這個視窗的 `set` 設定，不用打兩次；**這個視窗不能關**，關掉兩顆都會停

不想串接的話，也可以分開打：mix_exp8 這行跑完，再打 mix_exp9 那行（**只有 `--exp-name` 和 `--seg-weight` 不一樣**）。

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp8 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --seg-weight 0.5 --gpu 0
```

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp9 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --seg-weight 5 --gpu 0
```

🔴 **mix_exp8、mix_exp9 不能同時跑**：一顆實際就要約 12.2 GB，兩顆加起來超過 24 GB
（不像 mix_exp6、7 那樣各設 0.42 就能一起跑）。一顆跑完再跑下一顆。

**第一步會先自我檢查**，log 裡要看到這行，沒看到就是程式有問題，先停下來跟我說：

```
[v] 第一步檢查：形變場、搬移結果跟 VxmDense.forward 一致
```

**log 的樣子**：括號裡多了第三項（標籤項），例如

```
epoch: 0001  step: 1/100     time: 3.10 sec  loss: -0.429  (-0.084, 0.000000, -0.345)
                                                        總計    影像項  平滑項    標籤項
```

標籤項 = γ ×（−平均 Dice），所以 γ = 0.5 時大約 −0.35、γ = 5 時大約 −3.5，**負的、而且比影像項大很多是正常的**。
⚠️ 訓練過程中標籤項不一定會一直變小：標籤搬過去會被線性內插弄模糊，形變越大越模糊。挑 epoch 還是看驗證集的 Dice。

**要跑多久**：筆電量起來每步只比 mix_exp6 慢約 6%，另外每步要多讀一份標籤；mix_exp6 單獨跑是每步約 1.8 秒，
估計**每步約 2 秒、一顆約 14 小時，兩顆接著跑約 28 小時**。 以第一個 epoch 印出來的 `time:` 為準（看第 2 步以後）。
⚠️ 每步**超過 8 秒**就不對勁，先停下來跟我說，**不要自己改參數**。

**中斷了怎麼辦**：新視窗先 `cd`、啟動虛擬環境、**再打一次 `set` 那行**，然後在原本那行最後加 `--resume`，會從最後一個 `.pt` 接著跑。
用串接那行的話，看停在哪一顆（`models\mix_exp9\` 裡還沒有 `.pt` 就是停在 mix_exp8）：

停在 mix_exp8（mix_exp8 那半段加 `--resume`，跑完一樣會接著開始 mix_exp9）：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp8 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --seg-weight 0.5 --gpu 0 --resume && python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp9 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --seg-weight 5 --gpu 0
```

停在 mix_exp9（mix_exp8 已經跑完，只接著跑 mix_exp9）：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp9 --image-loss ncc --lambda 1.0 --epochs 250 --int-steps 7 --int-downsize 1 --seg-weight 5 --gpu 0 --resume
```

---

## 3. 訓練完 → 用驗證集挑 epoch（兩顆都要做）

🔴 **不要拿 test 挑 epoch。** test 只能跑一次。下面以 mix_exp8 為例，mix_exp9 把名字換掉再做一次。

```bat
python ASD\test_dice.py --model-dir models\mix_exp8 --test-dir data\mixed_preprocessed_v2\val --exp-name mix_exp8 --step 5 --gpu 0
```

```bat
copy models\mix_exp2\dice_baseline.csv models\mix_exp8\
copy models\mix_exp2\dice_baseline_val.csv models\mix_exp8\
python ASD\plot_dice_curve.py --model-dir models\mix_exp8 --label "速度場 權重1 標籤0.5"
python ASD\plot_loss_curve.py --logs log\mix_exp8.txt log\mix_exp6.txt --labels 用標籤 不用標籤 --out models\mix_exp8\loss_exp8_vs_exp6.png
```

- `test_dice.py` 跟以前一樣：測試時**只用影像**，標籤只拿來算分數
- `plot_loss_curve.py` 讀得懂三項，有標籤項的會多畫第四格（10-04 更新，要 `git pull` 過）
- ⚠️ 總 loss 跟 mix_exp6 不能比（目標函數不一樣），影像項、平滑項可以比

**挑法**：取 `dice_mean` 最大的那個 epoch。

---

## 4. test 評估（只跑一次）

把 `NNNN` 換成第 3 步挑出來的 epoch：

```bat
python ASD\test_dice.py --model models\mix_exp8\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_exp8 --gpu 0
```

mix_exp9 同樣做一次。

---

## 5. 要比的數字、怎麼判讀

| | mix_exp6 | **mix_exp8** | **mix_exp9** |
|---|---|---|---|
| 標籤權重 γ | 0（不用標籤）| 0.5 | 5 |
| test Dice | 0.8051 | **？** | **？** |
| 擠爆比例 | < 0.001% | **？** | **？** |

👉 **Dice**：
- **mix_exp8 明顯比 mix_exp6 高** → 訓練時用標籤有用，跟論文一致
- **mix_exp9 又比 mix_exp8 高** → 標籤項可以再加重；**比 mix_exp8 低** → 0.5 附近就夠了

👉 **擠爆比例**：論文的位移場版加了標籤，擠爆會跟著變多（γ 越大越多）。我們是速度場，看是不是還接近 0。

⚠️ **報告時要講清楚**：算 Dice 的 30 個結構，訓練時看過（看的是別的受試者的標籤，test 的人沒看過）。
跟完全不用標籤的模型比，是「多用了一種資訊」，不是同一種方法調得比較好。

帶回來之後逐人配對（51 位）、逐結構我來算。

---

## 6. 視覺化（挑好 epoch 之後）

跟之前幾顆同樣四位。兩行都要把 `NNNN` 換掉，可以一起貼：

```bat
for %s in (T054 D031 VNT045 sub-0043) do python ASD\visualize_dice.py --model models\mix_exp8\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_exp8\vis_%s --gpu 0
for %s in (T054 D031 VNT045 sub-0043) do python draw-img\visualize_reg_ixi.py --model models\mix_exp8\NNNN.pt --atlas IXI\atlas_mni152_09c_v3.npz --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_exp8\vis_%s --gpu 0
```

這步也可以不做，把 `.pt` 帶回來我這邊跑。

---

## 7. 收尾

挑好 epoch、test 跑完之後，只留最佳那顆 `.pt`，其他刪掉。要帶回主機的東西（兩顆各一份）：

```
models\mix_exp8\NNNN.pt                最佳 epoch 的權重
models\mix_exp8\dice_curve_val.csv     挑 epoch 的依據
models\mix_exp8\dice_NNNN.csv          test 結果
models\mix_exp8\*.png                  曲線圖
log\mix_exp8.txt                       訓練 stdout
log\mix_exp8_script.txt                當初下的指令
```

---

## 注意事項

- 🔴 **`--seg-weight`**：mix_exp8 是 0.5、mix_exp9 是 5。**不寫或寫 0 就變回一般訓練**（等於重跑 mix_exp6）
- 🔴 **`--int-downsize 1`、`--lambda 1.0`、`--image-loss ncc`**：跟 mix_exp6 一樣
- 🔴 訓練前要先打 `set PYTORCH_CUDA_ALLOC_CONF=...` 那行，換視窗要重打
- **`--atlas-seg` 不用給**：預設就是 `IXI\atlas_mni152_09c_v3_seg.npz`（FreeSurfer 那組）
- 筆電上做過的檢查（2026-10-04）：縮小一半的資料（訓練 20 位）、各訓練 1500 步，在沒看過的 6 位上：
  起點 0.690、不用標籤 0.717、**用標籤（γ = 0.5）0.774**，6 位每一位都比較高（手冊 §20.5）
- 📌 跑完把結果補進 `ASD\ASD相關手冊.md` 和 `models\README.md`
