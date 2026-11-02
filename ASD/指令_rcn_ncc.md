# RCN 換成局部 NCC（`models\rcn_vtn3_ncc`，cmd 版）

> **這份在做什麼**：第二步的第一個實驗（2026-11-02 使用者決定）。照原作的 RCN（VTN × 3）小腦白質低於 affine、被撐大，推測是全域相關係數配上偏亮的 MNI152 atlas 造成的（手冊 §31.11）。
> 這顆**只改相似度**：從整張影像算一個全域相關係數，改成每個 9×9×9 小方塊算的局部 NCC，其他都跟 `ASD/指令_reproduce.md` B 節一樣。
> 跑的機器：**AI**。指令全部是 **cmd（命令提示字元）** 語法。

| | B 節的 RCN（`orig_rcn_vtn3`）| 這顆（`rcn_vtn3_ncc`）|
|---|---|---|
| 相似度 | 全域相關係數 | **局部 NCC（窗格 9，算法同 VoxelMorph）** |
| 網路 | affine＋VTN × 3 | 相同 |
| 平滑權重 | `reg_factor` 1 | 相同（先不動，一次只改一項）|
| atlas | MNI152 | 相同 |
| 訓練資料 | 418 位 | 相同 |
| 訓練量 | batch 4、2.5 萬步（`--round 5000`）| 相同 |
| 資料增強、lr、優化器 | 照原作 | 相同 |
| 顯存 | 約 7 GB | 筆電試跑實際 7.23 GB、預留 7.69 GB |
| 時間 | 約 16 小時 | 估計 16～20 小時（局部 NCC 多一點計算，沒在 AI 實測）|
| 硬碟 | 25 顆約 9.5 GB，`last.pt` 約 1.2 GB | 相同 |

🔴 **改了 loss，就不是原作的 RCN 了**：論文上不能寫 `RCN (3-cascade VTN)`，要寫成 RCN-style，或取自己的名字。

---

## 開始前

```bat
cd /d D:\chengyun\TRY_voxelmorph
```

```bat
git pull
```

這次程式有改（`ASD/reproduce/rcn_torch.py`），一定要先 `git pull`。

---

## 1. 訓練

開一個新的 cmd：

```bat
cd /d D:\chengyun\TRY_voxelmorph
```

```bat
vxm_env\Scripts\activate.bat
```

還有別的程式在用 GPU 的話，先設顯存上限（只設在這個視窗）；單獨跑就不用設：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.4
```

```bat
python ASD\reproduce\train_rcn.py -b VTN -n 3 --train-dir data\mixed_preprocessed_v2\train --atlas IXI\atlas_mni152_09c_v3.npz --model-dir models\rcn_vtn3_ncc --round 5000 --save-every 1000 --net_args "sim='ncc'" -g 0
```

- 第二行開頭還是寫 `RCN（照原作）`，這是 log 的固定文字；**那一行最後要看到 `net_args {'sim': 'ncc'}`**，才代表真的換成局部 NCC
- 括號第一個數是相似度：這次是**負數**，一開始約 −0.04，之後往 −0.2 左右走（B 節那顆是正數）
- 括號第二個數是平滑項，一開始約 9，幾百步內會掉下來，正常（原作 VTN 的初始化）
- 跳出 `CUDA out of memory` 的話，上限調高 0.05 再跑一次

---

## 2. 用 val 挑（25 顆都掃）

```bat
python ASD\reproduce\eval_repro.py --method rcn --model-dir models\rcn_vtn3_ncc --step 1 --test-dir data\mixed_preprocessed_v2\val --gpu 0
```

取 `models\rcn_vtn3_ncc\dice_curve_val.csv` 裡 `dice_mean` 最大的步數。

---

## 3. test（只跑一次）

`NNNNNN` 換成挑出來的步數：

```bat
python ASD\reproduce\eval_repro.py --method rcn --model models\rcn_vtn3_ncc\NNNNNN.pt --test-dir data\mixed_preprocessed_v2\test --surface --gpu 0
```

---

## 跑完之後

把 `models\rcn_vtn3_ncc\` 裡的 `train_log.txt`、`dice_curve_val.csv`、`dice_NNNNNN.csv`、`surface_NNNNNN.csv` 和挑出來的 `NNNNNN.pt` 帶回筆電（要 `.pt` 才算得出每個結構的體積）。

**怎麼看結果**：

| | RCN（全域）| mix_exp3 | DWP | 這顆 |
|---|---|---|---|---|
| Dice | 0.7602 | 0.8062 | 0.8104 | ? |
| 小腦白質 Dice | 0.640 | 0.823 | 0.824 | ? |
| 小腦白質體積（atlas = 1）| 1.72 | 1.04 | 1.02 | ? |
| folding（每位）| 22 | 15,864 | 5,810 | ? |

- 小腦白質回到 0.82 左右、體積接近 1 倍 → 小腦問題來自全域相關係數，不是 VTN
- 小腦白質還是低、體積還是被撐大 → 問題在 VTN 或 RCN 的其他設定
- folding 暴增 → 局部 NCC 拉得比較用力，下一步再調平滑權重（`--net_args "sim='ncc', reg_factor=2"`）
