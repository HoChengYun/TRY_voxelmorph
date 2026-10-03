# mix_exp7 操作單（cmd 版）

> **這顆在問什麼**：老師 09-30 的紅字「速度場 Lambda 去調一下」。
> 速度場＋全尺寸，把平滑權重掃三個點：2（mix_exp5）、1（mix_exp6）、**0.5（這顆）**，
> 看 Dice 會不會繼續往上、擠爆會不會開始出現。
>
> | 實驗 | 版本 | `--int-steps` | `--int-downsize` | `--lambda` | 實際平滑權重 | test Dice | 擠爆 |
> |---|---|---|---|---|---|---|---|
> | mix_exp5 | 速度場 | 7 | 1 | 2.0 | 2 | 0.8026 | 0.000% |
> | mix_exp6 | 速度場 | 7 | 1 | 1.0 | 1 | （跑完補）| （跑完補）|
> | **mix_exp7** | **速度場** | **7** | **1** | **0.5** | **0.5** | **？** | **？** |
>
> 對照（位移場＋全尺寸）：權重 2 是 mix_exp4（0.8005、0.053%），權重 1 是 mix_exp3（0.8062、0.199%）。
>
> 跑的機器：**AI**，**等 mix_exp6 跑完再開始**（同一張顯卡）。指令全部是 **cmd（命令提示字元）** 語法。
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

確認 mix_exp6 已經跑完（`log\mix_exp6.txt` 最後一行是 `epoch: 0250  step: 100/100`），顯卡空出來了再開始。

---

## 1. 起跑前檢查（不會真的開始訓練）

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp7 --image-loss ncc --lambda 0.5 --epochs 250 --int-steps 7 --int-downsize 1 --gpu 0 --check-only
```

**顯存**：跟 mix_exp5 / mix_exp6 一樣約 **8.7 GB**（λ 不影響顯存）。

🔴 **`--int-downsize` 一定要寫 1**（預設是 2，漏寫就變回一半解析度）。
🔴 **`--lambda` 是 0.5**。實際平滑權重 = 0.5 × 1 = 0.5。

---

## 2. 訓練

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp7 --image-loss ncc --lambda 0.5 --epochs 250 --int-steps 7 --int-downsize 1 --gpu 0
```

會自動存 `log\mix_exp7.txt`、`log\mix_exp7_script.txt`、`models\mix_exp7\`，**不用自己加 `>` 導向**。

**要跑多久**：跟 mix_exp5 差不多，每步約 1.84 秒、約 13 小時。以第一個 epoch 印出來的 `time:` 為準。
⚠️ 每步**超過 5 秒**就是顯存塞不下。先停下來跟我說，**不要自己改參數**。

**中斷了怎麼辦**：用下面這行，會從最後一個 `.pt` 接著跑：

```bat
python ASD\run_train.py --train-dir data\mixed_preprocessed_v2\train --exp-name mix_exp7 --image-loss ncc --lambda 0.5 --epochs 250 --int-steps 7 --int-downsize 1 --gpu 0 --resume
```

---

## 3. 訓練完 → 用驗證集挑 epoch

🔴 **不要拿 test 挑 epoch。** test 只能跑一次。

```bat
python ASD\test_dice.py --model-dir models\mix_exp7 --test-dir data\mixed_preprocessed_v2\val --exp-name mix_exp7 --step 5 --gpu 0
```

接著把起點複製過來、畫圖（四行可以一起貼）：

```bat
copy models\mix_exp2\dice_baseline.csv models\mix_exp7\
copy models\mix_exp2\dice_baseline_val.csv models\mix_exp7\
python ASD\plot_dice_curve.py --model-dir models\mix_exp7 --label "速度場 全尺寸 平滑權重0.5"
python ASD\plot_loss_curve.py --logs log\mix_exp7.txt log\mix_exp6.txt log\mix_exp5.txt --labels 權重0.5 權重1 權重2 --out models\mix_exp7\loss_exp5_exp6_exp7.png
```

- 🔴 **不要加 `--baseline 0.6817`**：那個參數是給 test 的
- 🔴 最後一行**一定要 `--out`**
- ⚠️ 平滑項三顆不能直接比大小（λ 不一樣），**看中間那格影像項**

**挑法**：取 `dice_mean` 最大的那個 epoch。相鄰 epoch 的抖動約 ±0.003。
⚠️ 這顆的平滑限制最鬆，也要看 `jneg_pct` 那一欄：如果擠爆一路往上爬，挑 epoch 前先跟我說。

---

## 4. test 評估（只跑一次）

```bat
python ASD\test_dice.py --model models\mix_exp7\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --exp-name mix_exp7 --gpu 0
```

`NNNN` 換成第 3 步挑出來的 epoch。

---

## 5. 要比的數字、怎麼判讀

| 平滑權重 | 2（mix_exp5）| 1（mix_exp6）| **0.5（mix_exp7）** |
|---|---|---|---|
| test Dice | 0.8026 | ？ | **？** |
| 擠爆比例 | 0.000% | ？ | **？** |

👉 看三個點連起來的趨勢：
- **一路往上、擠爆還是接近 0** → 速度場可以放得更鬆，下一步試 0.25
- **1 最高、0.5 開始掉** → 甜蜜點在 1 附近
- **擠爆明顯出現** → 速度場「不擠爆」的保證也有極限（IXI exp4 權重 0.01 時到 2.26%）

---

## 6. 視覺化（可以不做，把 `.pt` 帶回來我這邊跑）

```bat
for %s in (T054 D031 VNT045 sub-0043) do python ASD\visualize_dice.py --model models\mix_exp7\NNNN.pt --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_exp7\vis_%s --gpu 0
for %s in (T054 D031 VNT045 sub-0043) do python draw-img\visualize_reg_ixi.py --model models\mix_exp7\NNNN.pt --atlas IXI\atlas_mni152_09c_v3.npz --test-dir data\mixed_preprocessed_v2\test --subject %s --out-dir models\mix_exp7\vis_%s --gpu 0
```

---

## 7. 收尾

只留最佳那顆 `.pt`，其他刪掉。要帶回主機的東西：

```
models\mix_exp7\NNNN.pt                最佳 epoch 的權重
models\mix_exp7\dice_curve_val.csv     挑 epoch 的依據
models\mix_exp7\dice_NNNN.csv          test 結果
models\mix_exp7\*.png                  曲線圖
models\mix_exp7\vis_*\                 視覺化（有做的話）
log\mix_exp7.txt                       訓練 stdout
log\mix_exp7_script.txt                當初下的指令
```

---

## 注意事項

- 🔴 **`--int-downsize 1`**、**`--lambda 0.5`**、**`--image-loss ncc`** 三個都要明寫
- 其他全部跟 mix_exp5 / mix_exp6 一樣：同一份資料、U-Net 預設寬度、250 epoch、學習率 1e-4
- 📌 跑完把結果補進 `ASD\ASD相關手冊.md` §20.5、`models\README.md`、CLAUDE.md「待辦 0」
