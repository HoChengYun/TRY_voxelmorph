# 2026-09-20 meeting 簡報產生器

520 顆資料 × 兩種版本（速度場／位移場）× 兩種標籤（FreeSurfer／tigerbx），加上交叉測試。
產出 `meeting報告/ASD_520顆與交叉測試_20260920.pptx`（14 頁）。

**投影片上的數字一律由 `gather.py` 從原始 CSV 算出，不手打。**
給老師看的版本，所以文字盡量少、圖盡量多、不用術語。

## 檔案

| 檔案 | 用途 |
|---|---|
| `gather.py` | 從 `models/*/dice_*.csv`、`mixed_manifest.json` 算出 `deck_data.json` |
| `make_charts.py` | 產生兩張總覽圖到 `models/deck_charts/` |
| `build.js` | pptxgenjs 產生器：版面、文字、表格、圖片 |
| `render.ps1` | 用 PowerPoint 把每頁轉成 PNG，做版面檢查 |

## 重建

```powershell
..\..\..\vxm_env\Scripts\python.exe make_charts.py     # -> models\deck_charts\*.png
..\..\..\vxm_env\Scripts\python.exe gather.py          # -> deck_data.json
node build.js deck.pptx
powershell -ExecutionPolicy Bypass -File render.ps1 -Pptx deck.pptx -OutDir render
copy deck.pptx ..\..\..\meeting報告\ASD_520顆與交叉測試_20260920.pptx
```

`build.js` 會去 `..\2026-09_mix_tigerbx\node_modules` 找 pptxgenjs，所以那個資料夾的
`npm install` 要先做過（已經做過了）。

## 需要的輸入（都在 `.gitignore` 擋掉的資料夾裡，clone 下來不會有）

- `models/{mix,tiger}_exp{2,3}/`：`dice_<epoch>.csv`、`dice_curve_val.csv`、`dice_baseline*.csv`、`vis_T054/`
- `models/mix_exp2/cross_mix_tiger_exp2_exp3/`：四個交叉評估 CSV + `cross_eval_2x2.png`
- `data/mixed_preprocessed_v2/mixed_manifest.json`

## 寫死在 `build.js` 裡、不是從 CSV 來的數字

- 論文 Table I 的折疊率（0.366%、ANTs SyN 0.185%、NiftyReg 0.793%）
- 上一次的資料量 286 顆
- 「以前挑 epoch 高估約 0.004」：手冊 §20.1 的實測

## 不進版控

`deck_data.json`、`deck.pptx`、`render/`
