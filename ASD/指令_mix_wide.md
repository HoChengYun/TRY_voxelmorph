# mix_wide 操作單

> **這顆在問什麼**：模型是不是太小了？
> 現在的 U-Net 只有 **301,411 個參數**。把每一層的通道數加倍 → **1,197,251 個（4.0 倍）**，
> 其他一模一樣，看 Dice 有沒有跟著上去。
>
> **對照組是 `mix_exp3`**（位移場版、λ=1.0、預設寬度，test Dice 0.8062）。
> 這兩顆的差別**只有 U-Net 的寬度**，所以結果可以直接歸因。
>
> 跑的機器：**AI**（這台筆電跑不動，25 秒/步）。
> 最後更新：2026-09-26

---

## 0. 開始前

```powershell
cd C:\Users\h4524\claude_cheng
.\vxm_env\Scripts\activate
```

需要 `data\mixed_preprocessed_v2\`（train 418 / val 51 / test 51）。
**它不在 git 裡**（`.gitignore` 擋掉整個 `data\`），要從別台機器搬過去，約 2 GB。
搬完先驗一次：

```powershell
python ASD\check_dataset.py --dirs data\mixed_preprocessed_v2 --check
```

---

## 1. 起跑前檢查（先跑這個，不會真的開始訓練）

```powershell
python ASD\run_train.py `
    --train-dir data\mixed_preprocessed_v2\train `
    --exp-name mix_wide `
    --image-loss ncc --lambda 1.0 --epochs 250 `
    --int-steps 0 --int-downsize 1 `
    --enc 32 64 64 64 --dec 64 64 64 64 64 32 32 `
    --gpu 0 --check-only
```

會檢查：atlas 在不在、資料夾有幾個 npz、GPU 有多少 VRAM、torch 版本。
**確認它印出來的 atlas 是 `atlas_mni152_09c_v3.npz`。**

### 🔴 先確認這台的顯示卡夠大

⚠️ `--check-only` **只會印出卡有多大，不會真的建模型試跑**。
它的警告門檻 7.5 GB 是為預設寬度設的，**加寬版要自己拿卡的容量對照下表**。

實測（全尺寸 192×224×192，一步 forward + backward + Adam 更新的峰值）：

| 設定 | 參數量 | 顯存 | 每步相對預設 |
|---|---|---|---|
| 預設寬度（mix_exp3）| 301,411 | 7.3 GB | 1.00× |
| 1.5 倍寬 | 675,027 | 10.3 GB | 1.42× |
| **2 倍寬（這顆）** | 1,197,251 | **13.5 GB** | 1.58× |

| 卡的顯存 | 怎麼做 |
|---|---|
| **16 GB 以上** | 照下面第 2 步直接跑 |
| 12 GB | 2 倍放不下，改跑 1.5 倍（見下）|
| 8 GB | 兩個都跑不了 |

（13.5 GB 是張量實際佔用，CUDA 本身和 PyTorch 快取還會多吃一些，所以 16 GB 才穩。）

📌 **為什麼參數多 4 倍，顯存只多 1.85 倍**：參數本身才 18 MB，根本不佔空間。
吃顯存的是每一層算出來的**中間特徵圖**（backward 要用，得全部留著），
它的大小是「通道數 × 192×224×192」，**只跟通道數成正比 → 2 倍**。
參數是「輸入通道 × 輸出通道 × 3×3×3」，兩邊都加倍才會變 4 倍。

### 卡不夠大：改跑 1.5 倍寬

```powershell
python ASD\run_train.py `
    --train-dir data\mixed_preprocessed_v2\train `
    --exp-name mix_wide15 `
    --image-loss ncc --lambda 1.0 --epochs 250 `
    --int-steps 0 --int-downsize 1 `
    --enc 24 48 48 48 --dec 48 48 48 48 48 24 24 `
    --gpu 0
```

這**還是單一變因**（跟 mix_exp3 只差寬度），只是加得比較少。
**名字一定要改成 `mix_wide15`**，不要跟 2 倍那顆混在一起。
下面第 3～6 步的 `mix_wide` 全部換成 `mix_wide15`。

### ❌ 不要用 `--int-downsize 2` 省記憶體

位移場版（`--int-steps 0`）下，**它完全不會改到網路**：
`voxelmorph\torch\networks.py:157` 寫的是 `resize = int_steps > 0 and int_downsize > 1`，
`int_steps` 是 0，縮放層根本不會建立。實測加了之後還是 13.5 GB，一點都沒省。

它唯一的效果在 `train.py:139`：把平滑項乘上 `int_downsize`，
**等於偷偷變成 `--lambda 2.0`**，也就是 mix_exp4 的設定。
這樣「加寬」和「平滑權重加倍」混在一起，結果就沒辦法歸因了。

---

## 2. 訓練

```powershell
python ASD\run_train.py `
    --train-dir data\mixed_preprocessed_v2\train `
    --exp-name mix_wide `
    --image-loss ncc --lambda 1.0 --epochs 250 `
    --int-steps 0 --int-downsize 1 `
    --enc 32 64 64 64 --dec 64 64 64 64 64 32 32 `
    --gpu 0
```

包裝腳本會自動處理掉這些，**不用自己加 `>` 導向**：

| 它會做的事 | 位置 |
|---|---|
| 存 stdout | `log\mix_wide.txt` |
| 存當初下的指令 | `log\mix_wide_script.txt` |
| 存權重 | `models\mix_wide\` |
| 帶 atlas | 預設就是 `IXI\atlas_mni152_09c_v3.npz` |

**要跑多久**：mix_exp3 在 AI 上是每步 1.49 秒 → 250 epoch 約 10.3 小時。
加寬版實測每步慢 1.58 倍 → **約 16 小時**（1.5 倍寬約 15 小時）。
這個倍數是在筆電上量的，AI 的卡不同可能有出入，**以第一個 epoch 印出來的 `time:` 為準**。

**中斷了怎麼辦**：同一行指令加 `--resume`，會從最後一個 `.pt` 接著跑。

---

## 3. 訓練完 → 用驗證集挑 epoch

🔴 **不要拿 test 挑 epoch。** test 只能跑一次。

```powershell
python ASD\test_dice.py `
    --model-dir models\mix_wide `
    --test-dir data\mixed_preprocessed_v2\val `
    --exp-name mix_wide --step 10 --gpu 0
```

輸出 `models\mix_wide\dice_curve_val.csv`（每 10 個 epoch 一列：Dice + 折疊率）。
畫成圖：

```powershell
python ASD\plot_dice_curve.py --model-dir models\mix_wide --baseline 0.6817 --label "寬度 2 倍"
python ASD\plot_loss_curve.py --logs log\mix_wide.txt log\mix_exp3.txt --labels 加寬 預設寬度
```

（`--baseline 0.6817` 是**驗證集**的起點；test 的起點是 0.6882，兩個不要搞混。
第二行順便把 mix_exp3 的 loss 疊上來，可以直接看兩顆收斂得一不一樣。）

**挑法**：取 `dice_mean` 最大的那個 epoch。
⚠️ 相鄰 epoch 的抖動有 ±0.003，曲線平的時候別太在意差 0.001 的名次。
⚠️ 順便看折疊率有沒有一路往上爬 —— 會的話代表形變開始硬凹。

---

## 4. 起點（baseline）

只做線性對位、還沒過模型的 Dice。**跟 mix_exp3 同一批 test，所以可以直接複製過來**：

```powershell
copy models\mix_exp2\dice_baseline.csv models\mix_wide\dice_baseline.csv
```

想自己重算也可以（結果會一樣，因為跟模型無關）：

```powershell
python ASD\test_dice.py --baseline `
    --test-dir data\mixed_preprocessed_v2\test `
    --exp-name mix_wide --gpu 0
```

---

## 5. test 評估（只跑一次）

把 `NNNN` 換成第 3 步挑出來的 epoch：

```powershell
python ASD\test_dice.py `
    --model models\mix_wide\NNNN.pt `
    --test-dir data\mixed_preprocessed_v2\test `
    --exp-name mix_wide --gpu 0
```

輸出 `models\mix_wide\dice_NNNN.csv`（逐人、逐結構）。

**要比的數字**（對照 `mix_exp3`）：

| | mix_exp3 | mix_wide |
|---|---|---|
| 參數量 | 301,411 | 1,197,251 |
| 起點 Dice | 0.6882 | 同一批，一樣 |
| test Dice | **0.8062** | ？ |
| 模型貢獻 | +0.118 | ？ |
| 折疊率 | 0.199% | ？ |

👉 **判讀**：Dice 沒動 → 容量不是瓶頸，是資料或設定的限制。
Dice 上去了 → 值得再往更大的模型試。
Dice 沒動但折疊率變高 → 模型多出來的容量拿去亂捏了，更該調平滑權重而不是加寬。

---

## 6. 視覺化（挑好 epoch 之後）

跟其他實驗用同一批受試者，才好並排比較：

```powershell
python ASD\visualize_dice.py `
    --model models\mix_wide\NNNN.pt `
    --test-dir data\mixed_preprocessed_v2\test `
    --subject T054 --out-dir models\mix_wide\vis_T054 --gpu 0

python draw-img\visualize_reg_ixi.py `
    --model models\mix_wide\NNNN.pt `
    --atlas IXI\atlas_mni152_09c_v3.npz `
    --test-dir data\mixed_preprocessed_v2\test --subject T054 `
    --out-dir models\mix_wide\vis_T054 --gpu 0
```

⚠️ `visualize_reg_ixi.py` **不給 `--subject` 會從 test 隨機挑一位**，跟別的實驗就對不起來了。

---

## 7. 收尾

```powershell
# 只留最佳的那顆 .pt，其他刪掉（250 個檔案會佔很多空間）
# 先確認 NNNN 是對的，再刪
```

要搬回主機的東西（`models\mix_wide\` 底下）：

```
NNNN.pt                最佳 epoch 的權重
dice_curve_val.csv     挑 epoch 的依據
dice_NNNN.csv          test 結果
dice_baseline.csv      起點
*.png                  曲線圖與視覺化
log\mix_wide.txt       訓練 stdout
log\mix_wide_script.txt 當初下的指令
```

---

## 注意事項

- 🔴 **`--atlas-seg` 不用換。** 這是 FreeSurfer 那組，用預設的
  `atlas_mni152_09c_v3_seg.npz`。（只有 tigerbx 那組才要換，忘了會安靜地低掉約 0.14）
- 🔴 **`--image-loss` 預設是 `mse`**，上面每一行都明寫了 `ncc`，不要漏掉。
- 🔴 **實際的平滑權重 = λ × int-downsize**。這裡是 1.0 × 1 = **1**，跟 mix_exp3 一樣。
  而且位移場版（`--int-steps 0`）下，`--int-downsize` **只會改平滑權重，完全不動網路**
  （見第 1 步最後一段）。
- 📌 跑完把結果補進 `ASD\ASD相關手冊.md` 和 `models\README.md`。
