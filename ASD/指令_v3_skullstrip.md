# v3 三種去頭殼各練一顆：操作單（cmd 版）

> **這三顆在問什麼**（老師 10/29 紅字 p8）：標籤固定用 FreeSurfer（aparc+aseg），**只換去頭殼工具**，
> 換工具之後 VoxelMorph 配準得好不好。
>
> - 三組的底圖都是 FreeSurfer 的 `orig.mgz`，影像 = orig × 各自的腦遮罩，**只差遮罩**（`data\skullstrip_compare\README.md`）
> - 前處理三組都一樣：`--n4`、各自算 affine 對到 MNI152（11-04 使用者決定各自算）
> - 切分三組共用：`data\skullstrip_compare\split_v3.json`，train 1061 / val 131 / test 131
> - 模型：**VoxelMorph 官方原版 `train.py`，一行不改**；設定同 mix_exp3（位移場、局部 NCC、λ 1）
> - 🔴 **訓練量 500 輪（5 萬步）**（11-17 使用者）：mix_exp3 第 250 輪還在爬，加上訓練資料變成 2.5 倍
>
> | 實驗名 | 去頭殼 | 資料 |
> |---|---|---|
> | `v3_freesurfer_exp3` | FreeSurfer（`brainmask.mgz`）| `data\v3_freesurfer_preprocessed\` |
> | `v3_synthstrip_exp3` | SynthStrip | `data\v3_synthstrip_preprocessed\` |
> | `v3_tigerbx_exp3` | tigerbx | `data\v3_tigerbx_preprocessed\` |
>
> 跑的機器：**AI**。指令全部是 **cmd** 語法。**一次只跑一顆**（三顆依序）。
> v3 的資料、標籤、test 都跟 mixed_v2 不同，**數字不能跟 mix_exp3 等舊模型比**，只比這三顆彼此。
> 最後更新：2026-11-17

---

## cmd 要注意的三件事

1. **每個指令都是一整行**：整行複製、貼上、按 Enter。看起來換行的地方是畫面自動折行
2. **換磁碟要用 `cd /d`**
3. 虛擬環境啟動成功的話，提示字元前面會多一個 `(vxm_env)`

---

## 0. 筆電這邊先做的（我做，AI 不用跑）

- [x] 三組前處理跑完（`log\v3_preprocess_<組>.txt` 最後一行 `完成！`；tigerbx 後段在 `..._tigerbx_part2.txt`），每組 train 1061 / val 131 / test 131（11-18）
- [x] 產生指紋檔：`check_dataset.py --write-manifest`（三個資料夾各一個 `dataset_manifest.json`；三組各 1323/1323 通過，11-18）。會印「沒有 subjects.txt，跳過對帳」：v3 不是從 fs_stats 來的，正常
- [ ] 要搬到 AI 的東西（共約 30 GB：FreeSurfer 11、SynthStrip 9.9、tigerbx 9.5）：

```
筆電（來源）                                                        →  AI（放這裡）
D:\MyHome\MRI\Cdisk_voxelmorph\v3_freesurfer_preprocessed\           →  D:\chengyun\TRY_voxelmorph\data\v3_freesurfer_preprocessed\
D:\MyHome\MRI\Cdisk_voxelmorph\v3_synthstrip_preprocessed\           →  D:\chengyun\TRY_voxelmorph\data\v3_synthstrip_preprocessed\
D:\MyHome\MRI\Cdisk_voxelmorph\v3_tigerbx_preprocessed\              →  D:\chengyun\TRY_voxelmorph\data\v3_tigerbx_preprocessed\
C:\Users\h4524\claude_cheng\IXI\atlas_mni152_09c_v3_seg_aparc+aseg.npz →  D:\chengyun\TRY_voxelmorph\IXI\
```

三個資料夾整個搬（含 split.json、dataset_manifest.json）；atlas 那個檔 `IXI\` 不進 git，要手動複製。筆電的資料 11-18 起都在 D 槽（`CLAUDE.md`），AI 上照舊放 repo 底下的 `data\`，下面的指令都不用改。

---

## 1. 開始前（AI）

```bat
cd /d D:\chengyun\TRY_voxelmorph
vxm_env\Scripts\activate.bat
git pull
```

檢查資料有沒有搬完整（每組約 10 分鐘，算 sha256）：

```bat
python ASD\check_dataset.py --check --dirs data\v3_freesurfer_preprocessed data\v3_synthstrip_preprocessed data\v3_tigerbx_preprocessed
```

（會印「沒有 subjects.txt，跳過對帳」，正常；要看的是最後一行「全部通過」。）

最後要印 `[v] 全部通過。` 才往下。出現 `🔴` 就先停下來跟我說（以前搬過只複製到 191／258 個的情況）。
「沒有 subjects.txt，跳過對帳」那行是正常的（v3 的名單在 split.json 裡）。

---

## 2. 起跑前檢查（不會真的開始訓練）

```bat
python ASD\run_train.py --train-dir data\v3_freesurfer_preprocessed\train --exp-name v3_freesurfer_exp3 --image-loss ncc --lambda 1.0 --epochs 500 --int-steps 0 --int-downsize 1 --gpu 0 --check-only
```

確認：訓練資料 **1061** 個、atlas 是 `atlas_mni152_09c_v3.npz`、epochs **500**。

🔴 最容易錯的四個參數：

| 參數 | 要寫 | 漏寫會變成 |
|---|---|---|
| `--int-steps` | **0**（位移場）| 預設 7 = 速度場 |
| `--int-downsize` | **1** | 預設 2 = 平滑權重偷偷變 2 倍 |
| `--lambda` | **1.0** | — |
| `--epochs` | **500** | 預設 250 |

---

## 3. 訓練（三顆依序）

**建議：一行跑完三顆**（前一顆正常結束才會接下一顆；中途出錯會停在那顆）：

```bat
python ASD\run_train.py --train-dir data\v3_freesurfer_preprocessed\train --exp-name v3_freesurfer_exp3 --image-loss ncc --lambda 1.0 --epochs 500 --int-steps 0 --int-downsize 1 --gpu 0 && python ASD\run_train.py --train-dir data\v3_synthstrip_preprocessed\train --exp-name v3_synthstrip_exp3 --image-loss ncc --lambda 1.0 --epochs 500 --int-steps 0 --int-downsize 1 --gpu 0 && python ASD\run_train.py --train-dir data\v3_tigerbx_preprocessed\train --exp-name v3_tigerbx_exp3 --image-loss ncc --lambda 1.0 --epochs 500 --int-steps 0 --int-downsize 1 --gpu 0
```

想一顆一顆跑的話，就是上面用 `&&` 隔開的三段，各自一行。

包裝腳本會自己存 `log\<實驗名>.txt`、`log\<實驗名>_script.txt`，權重存在 `models\<實驗名>\`，**不用自己加 `>`**。

**要跑多久**：mix_exp3 在 AI 上每步約 1.5 秒 → 一顆 5 萬步約 **21 小時**，三顆約 **2.6 天**。以第一個 epoch 印出來的 `time:` 為準。
⚠️ 每步**超過 5 秒**就是顯存塞不下，先停下來跟我說，不要自己改參數。

**中斷了怎麼辦**：哪一顆斷了，就用那一顆的指令最後加 `--resume`（會從最後一個 `.pt` 接著跑），例如：

```bat
python ASD\run_train.py --train-dir data\v3_synthstrip_preprocessed\train --exp-name v3_synthstrip_exp3 --image-loss ncc --lambda 1.0 --epochs 500 --int-steps 0 --int-downsize 1 --gpu 0 --resume
```

⚠️ `--resume` 會重開 `log\<實驗名>.txt`（覆蓋前面那段 log）。斷掉時先把那個 log 改名備份再續跑。

---

## 4. 用驗證集挑 epoch（每顆練完就可以做，不用等三顆都完）

> 🔴 **這一步要先 `git pull` 拿到 `test_dice.py --merge-cortex`**（11-17 已加，AI 上要 `git pull` 才有）。
> v3 的標籤是 aparc+aseg：皮質分成 68 區（1000～1999、2000～2999），沒有 3／42。
> 不併回去的話，論文 30 個結構裡的左右皮質會**安靜地被跳過**，Dice 變成 28 個結構的平均。

三組各跑一次起點（affine，不用模型）＋整條曲線。以 FreeSurfer 組為例（另兩組把三處 `freesurfer` 換掉）：

```bat
python ASD\test_dice.py --baseline --test-dir data\v3_freesurfer_preprocessed\val --exp-name v3_freesurfer_exp3 --atlas-seg IXI\atlas_mni152_09c_v3_seg_aparc+aseg.npz --merge-cortex --gpu 0
python ASD\test_dice.py --model-dir models\v3_freesurfer_exp3 --test-dir data\v3_freesurfer_preprocessed\val --exp-name v3_freesurfer_exp3 --atlas-seg IXI\atlas_mni152_09c_v3_seg_aparc+aseg.npz --merge-cortex --step 5 --gpu 0
```

- 輸出 `models\v3_freesurfer_exp3\dice_baseline_val.csv`、`dice_curve_val.csv`（100 個點）
- val 有 131 位（以前 51 位），曲線一顆大概要跑幾個小時
- **挑法**：取 `dice_mean` 最大的那個 epoch。三組各自挑

🔴 **不要拿 test 挑 epoch。** test 只跑一次。

---

## 5. test 評估（只跑一次）

把 `NNNN` 換成第 4 步挑出來的 epoch：

```bat
python ASD\test_dice.py --baseline --test-dir data\v3_freesurfer_preprocessed\test --exp-name v3_freesurfer_exp3 --atlas-seg IXI\atlas_mni152_09c_v3_seg_aparc+aseg.npz --merge-cortex --gpu 0
python ASD\test_dice.py --model models\v3_freesurfer_exp3\NNNN.pt --test-dir data\v3_freesurfer_preprocessed\test --exp-name v3_freesurfer_exp3 --atlas-seg IXI\atlas_mni152_09c_v3_seg_aparc+aseg.npz --merge-cortex --gpu 0
```

另兩組同樣換掉 `freesurfer` 與 `NNNN`。

---

## 6. 收尾、帶回來的東西

挑好 epoch、test 跑完之後，每組只留最佳那顆 `.pt`（500 個檔案很佔空間）。三組各帶回：

```
models\v3_<組>_exp3\NNNN.pt                 最佳 epoch 的權重
models\v3_<組>_exp3\dice_curve_val.csv      挑 epoch 的依據
models\v3_<組>_exp3\dice_baseline_val.csv   val 起點
models\v3_<組>_exp3\dice_baseline.csv       test 起點
models\v3_<組>_exp3\dice_NNNN.csv           test 結果
log\v3_<組>_exp3.txt                        訓練 stdout
log\v3_<組>_exp3_script.txt                 當初下的指令
```

配對比較、HD95、各結構、殘留體積 vs Dice（老師 p17）我在筆電這邊算。

---

## 怎麼判讀（帶回來之後）

- 三組的 **affine 起點本身就會不同**（11-04 示範：各自算 affine，tigerbx 的縮放每位都最大，affine 後 Dice 差 ≤ 0.014）。
  所以要同時看「起點 → 配準後」兩個數字，不能只看配準後
- 同一套設定重練一次 test 會差約 0.0006（§31.14.1）。三組差在 0.001 以內 → 打平
- 標籤一律是 FreeSurfer 的：三組的差異 = 去頭殼工具（遮罩）＋它造成的 affine 差異，沒有混進標籤差異（以前 tigerbx 那套連標籤都換了）

---

## 注意事項

- 🔴 `--int-steps 0`、`--int-downsize 1`、`--lambda 1.0`、`--epochs 500`、`--image-loss ncc` 都要明寫
- 三組的 `--atlas` 都是預設的 `IXI\atlas_mni152_09c_v3.npz`（前處理時對到的也是它）
- 訓練不讀標籤（`train.py` 只讀 `vol`），所以第 3 步不需要等 `--merge-cortex`
- 📌 跑完把結果補進 `ASD\ASD相關手冊.md`、`CLAUDE.md` 待辦 0c
