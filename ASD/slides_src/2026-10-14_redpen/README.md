# 2026-10-14 meeting 簡報產生器：回覆 09-30 老師的紅字

09-30 meeting 老師寫在 `meeting報告/ASD_520顆與交叉測試_20260920.pptx` 第 18、22、23、25 頁的五件事
（清單在 CLAUDE.md「待辦 0」、細節在手冊 §24）。
產出 `meeting報告/ASD_老師紅字回覆_20261014.pptx`（20 頁；mix_exp6、7 沒結果時少 2 頁、沒有 `folding_params.png`、`curve_wide.png`、`1014_six.png` 或 `1014_six_base.png` 時各少 1 頁）。**不會碰 09-20 那份**（上面有老師的紅字）。

**投影片上的數字一律由 `gather.py` 從原始 CSV 算出，不手打。**
給老師看的版本，所以文字盡量少、圖盡量多、不用術語。

| 頁 | 內容 | 圖 |
|---|---|---|
| 1–2 | 封面、一頁看完（五件事的結果表）| |
| 3–6 | ① 擠爆的位置（第 4 頁「不同設定」要先有 `folding_params.png`）| `folding_check/folding_views.png`、`folding_check/folding_params.png`、`1014_folding_regions.png`、`folding_check/folding_zoom_T054.png` |
| 7–10 | ② 速度場的平滑權重（9、10 頁只在 mix_exp6、7 都有結果時才出現）| `deck_charts/ablation.png`（09-20 那份的）、`1014_lambda.png`、`1014_lambda_struct.png`、`grid_lambda.png` |
| 11–14、16、17 | ③ 只算殘留旁邊的 Dice（第 12 頁散佈圖＋2×2 Dice 表；第 13 頁 test 頭頂殘留最多／最少各 3 位；第 14 頁三個位置的散佈圖＋Dice 表；第 16 頁顱底的 3 對 3）、是不是 FreeSurfer 畫太小（第 17 頁）| `1014_dilution.png`、`1014_six.png`、`1014_regions.png`、`1014_six_base.png`、`1014_top_example.png` |
| 15 | ④ 後腦杓（test 後腦杓殘留最多／最少各 3 位，跟第 13 頁同一個樣子）| `1014_six_back.png` |
| 18 | ⑤ 加寬＋速度場（2×2 表格：版本 × 寬度；右邊逐人配對；下面顯存設定前後每步時間）| （表格）|
| 19 | ⑤ 訓練過程：版本 × 寬度四顆的驗證集 Dice 與擠爆比例（mix_wide_vel 有結果、而且有圖才出現）| `deck_charts/curve_wide.png` |
| 20 | 下一步（訓練時也用標籤 mix_exp8／9、MRS）| |

📌 **③④ 段的規則（2026-10-05 跟使用者定案，細節手冊 §24.2）**：
- Dice 數值一律寫「起點 → 配準後」，兩個一樣大，不能只放大配準後（頭頂殘留多的人起點略高、後腦杓殘留多的 3 位起點反而低，
  只看配準後兩個方向都會被騙）。「模型貢獻」一律叫「Dice 進步多少」
- 殘留量用原始單位：頭頂、後腦杓是厚度（mm），顱底是最大一坨的體積（mm³）。相關係數、分組（170 人的 1/4、3/4 分位數）
  都直接用原始數字算（`gather.py` 的 `residue_mm`），跟以前「每批各自排名次」的版本幾乎一樣
- `1014_dilution.png`、`1014_regions.png`：橫軸殘留量、縱軸 Dice 進步多少，同一張圖裡每格縱軸刻度一樣（只是起點不同）
- `1014_six.png`（頭頂）、`1014_six_back.png`（後腦杓）、`1014_six_base.png`（顱底）同一個函式畫：test 殘留最多／最少各 3 位，每位一個切面。
  顱底的切面是穿過最大一坨殘留中心的矢狀面（每人位置不同），大字前面寫「旁邊結構」（顱底旁邊有腦幹，不能叫皮質）

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
.\vxm_env\Scripts\python.exe ASD\check_folding.py --models mix_exp6:0190 mix_exp7:0250 mix_wide_vel:0240 --gpu 0   # 速度場三顆的擠爆位置（位移場三顆 09-30 算過）
.\vxm_env\Scripts\python.exe ASD\check_folding.py --plot-only --views   # -> folding_views.png、folding_params.png（6 欄：左 3 顆速度場、右 3 顆位移場）
cd ASD\slides_src\2026-09-20_cross
..\..\..\vxm_env\Scripts\python.exe make_compare.py --set lambda   # -> curve_lambda / jacobian_lambda / grid_lambda.png（要先有各顆的 vis_T054）
..\..\..\vxm_env\Scripts\python.exe make_compare.py --set wide     # -> curve_wide / jacobian_wide / grid_wide.png（簡報只用 curve_wide）
cd ..\2026-10-14_redpen
node build.js deck.pptx
powershell -ExecutionPolicy Bypass -File render.ps1 -Pptx deck.pptx -OutDir render
copy deck.pptx ..\..\..\meeting報告\ASD_老師紅字回覆_20261014.pptx
```

🔴 **複製到 `meeting報告\` 之前，先確認那份沒被使用者改過**（`fc /b` 或比修改時間）。
使用者會在 PowerPoint 裡加字，覆蓋掉就沒了（09-30 那份就是這樣）。

`build.js` 會去 `..\2026-09_mix_tigerbx\node_modules` 找 pptxgenjs（跟 09-20 那份一樣）。

## 還沒跑完的實驗

mix_exp6、mix_exp7（10-04）、mix_wide_vel（10-05）都帶回來了，現在沒有 pending 的。
`gather.py` 找不到某顆的 test CSV（`models/<exp>/dice_<4 位數>.csv`）就標成 pending，簡報第 2 頁、平滑權重那頁、加寬＋速度場那頁顯示「跑中」、圖上標「跑中」框。
**結果帶回來放進 `models/<exp>/` 之後，照上面重建一次就會補上**，不用改程式。
一個資料夾有好幾份 test CSV 的話，用 `dice_curve_val.csv` 裡驗證集最好的那個 epoch。
mix_exp8／9（訓練時也用標籤）不在這份簡報的設定裡，只在「下一步」寫一行；結果要放進來得另外加頁。

## 寫死在程式裡、不是從 CSV 來的

- 每顆的設定（版本、實際平滑權重、寬度、積分解析度）：`gather.py` 的 `CFG`，照操作單
- 論文 Table I 的擠爆比例 0.366%（`make_charts.py`）
- 對照圖的兩位受試者 MRS0381-2、T054（跟 `check_top_residue.py --example` 同一組）
- 第 12 頁 2×2 表最後一列「起點就高」「進步差不多」「起點差不多」、第 15 頁「兩組交錯在一起，分不出誰殘留多」、
  第 16 頁「這 6 位裡殘留多的進步還略多」「殘留大多在腦的前下方」這幾個字是照現在的數字寫的（數字本身從 CSV 算）
- 第 18 頁「約 33 GB／24 GB」：手冊 §23.7 的外插；「每步 10 幾秒 → 約 3 秒」：使用者 10-02 在 AI 上看到的（mix_exp6、7 同時跑）。
  10 幾秒那段**沒留在 log 裡**（帶回來的 log 是加了設定後從頭跑的，每步 3.0～3.3 秒）。
  加寬兩顆的每步秒數、小時數是 `gather.py` 從 `log/mix_wide.txt`、`log/mix_wide_vel.txt` 算的（log 沒帶回來時才退回寫死的「約 19 小時」）
- 第 20 頁「訓練時也用 FreeSurfer 標籤…標籤權重 0.5、5」：照 `ASD/指令_mix_exp8_9.md`

## 不進版控

`deck_data.json`、`deck.pptx`、`render/`
