# 2026-10-14 meeting 簡報產生器：回覆 09-30 老師的紅字

09-30 meeting 老師寫在 `meeting報告/ASD_520顆與交叉測試_20260920.pptx` 第 18、22、23、25 頁的五件事
（清單在 CLAUDE.md「待辦 0」、細節在手冊 §24）。
產出 `meeting報告/ASD_老師紅字回覆_20261014.pptx`（mix_exp6、7 有結果時 16 頁，沒有時 14 頁）。**不會碰 09-20 那份**（上面有老師的紅字）。

**投影片上的數字一律由 `gather.py` 從原始 CSV 算出，不手打。**
給老師看的版本，所以文字盡量少、圖盡量多、不用術語。

| 頁 | 內容 | 圖 |
|---|---|---|
| 1–2 | 封面、一頁看完（五件事的結果表）| |
| 3–6 | ① 擠爆的位置（第 4 頁「不同設定」要先有 `folding_params.png`）| `folding_check/folding_views.png`、`folding_check/folding_params.png`、`1014_folding_regions.png`、`folding_check/folding_zoom_T054.png` |
| 7–10 | ② 速度場的平滑權重（9、10 頁只在 mix_exp6、7 都有結果時才出現）| `deck_charts/ablation.png`（09-20 那份的）、`1014_lambda.png`、`1014_lambda_struct.png`、`grid_lambda.png` |
| 11–13、15 | ③ 只算殘留旁邊的 Dice、是不是 FreeSurfer 畫太小 | `1014_dilution.png`、`1014_regions.png`、`1014_top_example.png` |
| 14 | ④ 後腦杓 | `1014_back_example.png` |
| 16 | ⑤ 加寬＋速度場 | （表格）|
| 17 | 下一步 | |

頁碼由 `build.js` 開頭的 `ORDER` 決定，內文引用的頁碼（第 2 頁、最後一頁）會跟著算；做出來的頁數跟 `ORDER` 對不上會直接報錯。

## 檔案

| 檔案 | 用途 |
|---|---|
| `gather.py` | 從 `models/*/dice_*.csv`、`models/folding_check/`、`models/skullstrip_check/` 算出 `deck_data.json` |
| `make_charts.py` | 產生 `models/deck_charts/1014_*.png` |
| `build.js` | pptxgenjs 產生器：版面、文字、表格、圖片 |
| `render.ps1` | 用 PowerPoint 把每頁轉成 PNG，做版面檢查。PowerPoint 本來就開著的話不會把它關掉 |

## 重建

```powershell
cd ASD\slides_src\2026-10-14_redpen
..\..\..\vxm_env\Scripts\python.exe gather.py          # -> deck_data.json
..\..\..\vxm_env\Scripts\python.exe make_charts.py     # -> models\deck_charts\1014_*.png（要先有 deck_data.json）
cd ..\..\..
.\vxm_env\Scripts\python.exe ASD\check_skullstrip.py --show back --n 1 --csv models\skullstrip_check\skullstrip_all520.csv --dice models\mix_exp3\dice_0240.csv --baseline models\mix_exp2\dice_baseline.csv --out models\deck_charts\1014_back_example.png
.\vxm_env\Scripts\python.exe ASD\check_folding.py --models mix_exp6:0190 mix_exp7:0250 --gpu 0   # 速度場兩顆的擠爆位置（位移場三顆 09-30 算過）
.\vxm_env\Scripts\python.exe ASD\check_folding.py --plot-only --views   # -> folding_views.png、folding_params.png
cd ASD\slides_src\2026-09-20_cross
..\..\..\vxm_env\Scripts\python.exe make_compare.py --set lambda   # -> curve_lambda / jacobian_lambda / grid_lambda.png（要先有各顆的 vis_T054）
cd ASD\slides_src\2026-10-14_redpen
node build.js deck.pptx
powershell -ExecutionPolicy Bypass -File render.ps1 -Pptx deck.pptx -OutDir render
copy deck.pptx ..\..\..\meeting報告\ASD_老師紅字回覆_20261014.pptx
```

🔴 **複製到 `meeting報告\` 之前，先確認那份沒被使用者改過**（`fc /b` 或比修改時間）。
使用者會在 PowerPoint 裡加字，覆蓋掉就沒了（09-30 那份就是這樣）。

`build.js` 會去 `..\2026-09_mix_tigerbx\node_modules` 找 pptxgenjs（跟 09-20 那份一樣）。

## 還沒跑完的實驗

mix_exp6、mix_exp7、mix_wide_vel 在 AI 上跑。`gather.py` 找不到它們的 test CSV（`models/<exp>/dice_<4 位數>.csv`）
就標成 pending，簡報第 2、7、13 頁顯示「跑中」、圖上標「跑中」框。
**結果帶回來放進 `models/<exp>/` 之後，照上面重建一次就會補上**，不用改程式。
一個資料夾有好幾份 test CSV 的話，用 `dice_curve_val.csv` 裡驗證集最好的那個 epoch。

## 寫死在程式裡、不是從 CSV 來的

- 每顆的設定（版本、實際平滑權重、寬度、積分解析度）：`gather.py` 的 `CFG`，照操作單
- 論文 Table I 的擠爆比例 0.366%（`make_charts.py`）
- 對照圖的兩位受試者 MRS0381-2、T054（跟 `check_top_residue.py --example` 同一組）
- 第 15 頁「約 33 GB／24 GB／約 19 小時／26 小時」：手冊 §23.7 與 `ASD/指令_mix_wide_vel.md` 的估計；
  「每步 10 幾秒 → 約 3 秒」：使用者 10-02 在 AI 上看到的（mix_exp6、7 同時跑）。
  10 幾秒那段**沒留在 log 裡**（帶回來的 log 是加了設定後從頭跑的，每步 3.0～3.3 秒）

## 不進版控

`deck_data.json`、`deck.pptx`、`render/`
