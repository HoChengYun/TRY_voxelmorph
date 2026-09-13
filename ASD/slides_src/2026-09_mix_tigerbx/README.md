# 2026-09 meeting 簡報產生器

三包混合訓練 × tigerbx 對照 × 跟論文比。
產出 `meeting報告/ASD延伸實驗_混合訓練與tigerbx_v2.pptx`（29 頁，含備忘稿）。

配色沿用上一份簡報。**投影片上的實驗數字一律由 `gather.py` 從 CSV 算出，不手打。**
頁碼參照（第 2 頁總覽表等）由 `build.js` 依頁序自動填，加頁不會對錯頁。

## 檔案

| 檔案 | 用途 |
|---|---|
| `build.js` | pptxgenjs 產生器：版面、文字、圖表、備忘稿 |
| `gather.py` | 從 `models/*/dice_*.csv`、`mixed_manifest.json`、`demographics.tsv` 算出 `deck_data.json` |
| `bench.py` | 量推論時間 → `bench.json`（GPU 暖機 2 次後量 28 顆；CPU 單線程量 3 顆）|
| `bench.json` | 2026-09-13 在 RTX 4060 Laptop 上量的結果。**重跑 `bench.py` 會覆寫** |
| `render.ps1` | 用 PowerPoint 把每頁轉成 PNG，做版面檢查 |

## 重建

在這個資料夾裡：

```powershell
npm install                                      # 第一次才要（pptxgenjs 4.0.1）
..\..\..\vxm_env\Scripts\python.exe gather.py    # → deck_data.json
node build.js deck.pptx
powershell -ExecutionPolicy Bypass -File render.ps1 -Pptx deck.pptx -OutDir render
```

⚠️ 要在 **A 台**跑：`gather.py` 讀的 `models/`、`data/` 都被 `.gitignore` 擋掉，clone 下來是沒有的。
需要的檔案：

- `models/mix_exp1/`：`dice_curve.csv`、`dice_baseline.csv`、`dice_0230.csv`、`vis_*/`
- `models/tiger_exp1/`：`dice_curve.csv`、`dice_baseline.csv`、`dice_0240.csv`、`vis_*/`
- `models/asd_exp1/dice_0190.csv`
- `data/mixed_preprocessed_v1/mixed_manifest.json`
- `data/{ASD,DGM,VNT}_data/fs_stats/demographics.tsv`

## 不在 CSV 裡、寫死在程式中的數字

- `gather.py` 的 `dup_dist` / `dup_pairs`：`find_duplicate_scans.py` 的實測結果
- `build.js` 的論文數字：Table I、Table II 直接抄論文；**Fig. 7 的 λ 數字是從圖上量的（約 ±0.001）**
- `build.js` 的資料把關頁（排除表、DICOM 那頁）：來自 FreeSurfer 端的 DICOM 核對

## 不進版控

- `deck_data.json`：含逐人年齡，公開 repo 不放（`gather.py` 隨時可以重算）
- `node_modules/`、產出的 `*.pptx`、`render/`
