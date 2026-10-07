# 2026-10-14 meeting 簡報產生器：回覆 09-30 老師的紅字

09-30 meeting 老師寫在 `meeting報告/ASD_520顆與交叉測試_20260920.pptx` 第 18、22、23、25 頁的五件事
（清單在 CLAUDE.md「待辦 0」、細節在手冊 §24）。
產出 `meeting報告/ASD_老師紅字回覆_20261014.pptx`（37 頁；補充評估指標兩頁要先有 `surface_*.csv` 與 `1014_metric_*.png`、⑥ 架構修改六頁要先有 `1014_arch_*.png` 八張圖（`make_charts.py` 兩張、`make_arch.py` 六張）和第 0 步的 CSV、mix_exp6、7 沒結果時少 2 頁、`make_method.py` 的 8 張圖沒齊時少 4 頁、⑤ 補充的四張圖沒有時各少 1 頁、沒有 `folding_params.png`、`curve_wide.png`、`1014_six.png` 或 `1014_six_base.png` 時各少 1 頁）。**不會碰 09-20 那份**（上面有老師的紅字）。

**投影片上的數字一律由 `gather.py` 從原始 CSV 算出，不手打。**
給老師看的版本，所以文字盡量少、圖盡量多。**2026-10-07 起全份（投影片文字與圖上的字）改用正式學術用語**，術語對照見下方 📌。

🔴 **Folding 跟論文比較一律用 voxel 數（2026-10-07 使用者決定）**：
- 論文文字（TMI 2019 §V-A-2）是「count all non-background voxels for which |J| ≤ 0」，但 Table I 標題寫明百分比的分母是**固定的 5.2 M voxel**（"for our volumes with 5.2 million voxels within the brain"）。我們的分母不同（atlas 非背景 1,867,705、整個影像 8,257,536），**百分比不能跟 0.366% 比**
- 跟論文比較：每位平均 folding voxel 數（都是 1 mm³），論文 VoxelMorph (CC) **19,077**。第 7、8、23 頁的論文虛線是 19,077，第 30 頁表格有一欄 voxel 數。我們的 voxel 數 = dice CSV 的 `jneg_pct` × 整個影像大小（精確值，`gather.py` 的 `models[e]['points']`）
- 簡報上的百分比（第 7、8、10、22、30 頁）分母是 atlas 非背景 voxel（`ASD/test_dice.py --surface` 的 `jneg_fg_pct`），只拿來比我們自己的模型。`gather.py` 把 `models[e]['jneg']` 換成這個，原本的值（分母：整個影像）留在 `jneg_all`；有模型缺 surface CSV 會直接報錯
- 第 4 頁欄標題、⑥ 的 Step 0 也只用 voxel 數
- 當天一度改成「分母：非背景」再跟 0.366% 比（mix_exp3 0.849%，看起來是論文的 2.3 倍），後來發現論文分母是 5.2 M 才改成比 voxel 數；用 voxel 數比，mix_exp3 16,420 跟論文 19,077 相當

📌 **正式用語（2026-10-07，使用者：「正式一點，公式 block 都很重要，不要太口語，你的中文也不要太口語」；
⑥ 先改，同天使用者同意 ①～⑤ 與下一步一起改）**：
- 擠爆 → folding（|J| ≤ 0）、folding voxel；速度場 → SVF；位移場 → displacement field（圖上簡寫 Displacement）；
  平滑權重 → λ；加寬 2 倍 → 2× width（預設寬度 → default width）；全尺寸／半解析度 → full-res.／half-res.
- 起點 → affine（「Dice 起點 → 配準後」改寫成「Dice（affine → 配準後）」，兩個數字照舊一起寫）；Dice 進步多少 → ΔDice
- 頭頂 → 顱頂；後腦杓 → 枕部；殘留旁邊的結構 → 殘留鄰近結構；沒切乾淨 → 去顱骨不完全；吸管 → column；格子 → voxel；
  水 → 腦脊髓液（CSF）；老師的算法 → 會議建議方法；訓練輪數 → epoch；驗證集 → validation set；跑中 → 訓練中
- 「沒差」改成寫 p 值：`gather.py` 的 `paired` 多算配對 Wilcoxon（λ 1 → 0.5：p = 0.18；2× width 下 SVF vs displacement：p = 0.17）
- 圖上的 ≤、≥、− 這三個字 Microsoft JhengHei 沒有：各支畫圖程式把 `font.family` 設成 `['Microsoft JhengHei', 'DejaVu Sans']`
  讓缺字自動改用 DejaVu Sans。⚠️ 字串裡有 `$...$`（mathtext）時這招無效、缺字會變成 ¤，所以 `make_method.py` 的 ≥、< 都寫在 `$\geq\tau$` 裡
- 投影片標題打不出下標（`z_top` 會變成底線），第 12～15 頁標題不放符號，符號在公式區塊裡定義

| 頁 | 內容 | 圖 |
|---|---|---|
| 1–2 | 封面、一頁看完（五件事的結果表）| |
| 3–6 | ① 擠爆的位置（第 4 頁「不同設定」要先有 `folding_params.png`）| `folding_check/folding_views.png`、`folding_check/folding_params.png`、`1014_folding_regions.png`、`folding_check/folding_zoom_T054.png` |
| 7–10 | ② 速度場的平滑權重（9、10 頁只在 mix_exp6、7 都有結果時才出現）| `1014_ablation.png`（09-20 那份 `ablation.png` 的正式用語版，`make_charts.py` 畫；09-20 的產生器不動）、`1014_lambda.png`、`1014_lambda_struct.png`、`grid_lambda.png` |
| 11 | ③ 老師的做法：只平均殘留旁邊的結構。表格一個結構一列：中文、FreeSurfer 名稱（標籤編號）、左／右占比、是否納入（`gather.py` 的 `residue_near`，2026-10-07 改）| （表格）|
| 12–15 | ③ 頭頂殘留厚度怎麼量（1/4～4/4）：每一步一頁，上面是圖、下面是編號公式 (1)～(5)＋「其中」符號說明（`make_method.py`）| `1014_method_{1..4}.png`、`1014_method_eq{1..4}.png` |
| 16–18、20、21 | ③ 第 16 頁散佈圖＋2×2 Dice 表；第 17 頁 test 頭頂殘留最多／最少各 3 位；第 18 頁三個位置的散佈圖＋Dice 表（附 FreeSurfer 結構名稱與標籤編號）；第 20 頁顱底的 3 對 3；第 21 頁是不是 FreeSurfer 畫太小 | `1014_dilution.png`、`1014_six.png`、`1014_regions.png`、`1014_six_base.png`、`1014_top_example.png` |
| 19 | ④ 後腦杓（test 後腦杓殘留最多／最少各 3 位，跟第 17 頁同一個樣子）| `1014_six_back.png` |
| 22 | ⑤ 加寬＋速度場（2×2 表格：版本 × 寬度；右邊逐人配對）。原本下方的「附記」（顯存設定前後每步時間）2026-10-07 使用者在 PowerPoint 刪掉，`build.js` 同步拿掉 | （表格）|
| 23 | ⑤ 訓練過程：版本 × 寬度四顆的驗證集 Dice 與擠爆比例，下面一行「不用再訓練更久」（`gather.py` 的 `plateau`）| `deck_charts/curve_wide.png` |
| 24 | ⑤ 每個結構：加寬的效果（兩個版本並排）、加寬後換版本（2026-10-06 加，以下同）| `1014_wide_struct.png` |
| 25 | ⑤ 越難對的人加寬幫越多：位移場、速度場並排＋起點最差／中間／最好的表（`gather.py` 的 `wide_diff`）| `1014_wide_difficulty.png` |
| 26 | ⑤ 訓練 loss：四顆的影像項、平滑項＋最後一輪的表（`gather.py` 的 `loss_final`）。平滑項不能跨版本比 | `1014_wide_loss.png` |
| 27 | ⑤ 視覺化（大圖）：T054 整片腦三個方向，加寬位移場 vs 加寬速度場，藍框＝下一頁放大的那一塊（使用者：「可以來大圖的嗎」）| `folding_check/folding_full_pair_T054.png` |
| 28 | ⑤ 視覺化：藍框那一塊放大，加寬位移場 vs 加寬速度場（`ASD\check_folding.py --zoom-pair` 兩張一起畫，要 GPU）| `folding_check/folding_zoom_pair_T054.png` |
| 29 | 補充評估指標之定義（2026-10-07 使用者：「把 HD95 和 SDlogJ 加進去，然後可以更新這次 meeting 簡報」）：上方示意圖（合成之 2D 例子）、下方式 (1)～(3)：HD95、SDlogJ、folding（分母：atlas 非背景；與論文比較用 voxel 數）（`make_metrics.py`）| `1014_metric_demo.png`、`1014_metric_eq.png` |
| 30 | 補充評估指標之結果：8 個模型 × Dice、HD95、SDlogJ、folding voxel 數（每位平均，虛線為論文 VoxelMorph (CC) 19,077）四格圖＋重點；affine 標在各格右上（2026-10-07 原本的表格改成圖；`gather.py` 的 `surface`、`surface_paired`；數值來自 `ASD/test_dice.py --surface`）| `1014_metrics.png`（`make_charts.py`）|
| 31 | ⑥ 架構修改方向：加入配準專用設計，而非更換 backbone（老師紅字之外，2026-10-06 使用者：「把改架構的進度也加進 10/14 簡報」）。左 Jian et al., WBIR 2024 Table 2（LPBA）、右 LUMIR 2024 test leaderboard；圖下一行資料來源 | `1014_arch_lit.png` |
| 32 | ⑥ Step 0：Test-time recursion（不重新訓練）。三顆模型 1、2、3 passes 並排＋右側重點（gather.py 的 `multipass`、`multipass_vs_wide`）＋式 (1) | `1014_arch_step0.png`、`1014_arch_step0_eq.png` |
| 33 | ⑥ Step 1：Cascaded registration（Zhao et al., ICCV 2019）。架構圖＋式 (2)(3) | `1014_arch_cascade.png`、`1014_arch_cascade_eq.png` |
| 34 | ⑥ Step 2 之結構：以 VoxelMorph U-Net 為基礎之三處修改（2026-10-07 使用者：「最後一頁不好想像，和 U-Net 本身架構有點搞混」）。左 VoxelMorph U-Net（直式 U）、右 Step 2 轉成同一方向，標出 ① dual-stream encoder ② m 之 skip 先 warp ③ 每層輸出形變；圖下一行說明與下一頁的對應 | `1014_arch_unet_vs_pyramid.png`（`make_arch.py`）|
| 35 | ⑥ Step 2：Coarse-to-fine（multi-resolution pyramid + warping）。架構圖＋式 (4)(5) | `1014_arch_pyramid.png`、`1014_arch_pyramid_eq.png` |
| 36 | ⑥ 實驗設計（2 × 2）與進度：2 × 2 表（沒結果的格子寫「待訓練」，mix_cascade_pyramid 寫「待前兩者結果」）＋參數量／訓練時間表（gather.py 的 `arch`）＋實作驗證、訓練順序 | （表格）|
| 37 | 下一步（訓練時也用標籤 mix_exp8／9、架構修改的順序、MRS）| |

📌 **⑥ 架構修改（第 31～35 頁）的寫法（2026-10-07 定案）**：使用者：「這個 block 有些太口語了」「由粗到細老師肯定不知道」
「正式一點，公式 block 都很重要，不要太口語」。因此 ⑥ 全部改成**正式學術用語＋架構圖＋編號公式 (1)～(5)**，每頁條列 2～4 行：
- 術語對照（使用者認可）：串兩顆 → Cascade（2 stages）、Stage 1／Stage 2；由粗到細 → Coarse-to-fine（multi-resolution pyramid）；
  先拉過去／接成一個 → warp／compose；兩張影像各自抽特徵 → Dual-stream encoder；同一顆多跑一次 → Test-time recursion（不重新訓練）；
  擠爆 → Folding（|J| ≤ 0）；原始／目標 → Moving（affine only）／Fixed（MNI152）
- 合成照 VoxelMorph 記號 $(m\circ\phi)(x)=m(\phi(x))$：先 $\phi_1$、再對 $m\circ\phi_1$ 估計 $\phi_2$，總形變 $\Phi=\phi_1\circ\phi_2$
  （位移 $U(x)=u_2(x)+u_1(x+u_2(x))$，跟 `ASD/arch.py` 一致）
- 架構圖與公式區塊由 `make_arch.py` 畫（公式樣式同 `make_method.py`）。原本 ⑥ 第 3 頁用 T054 真影像講解的 `1014_arch_explain.png` 已拿掉

📌 **③④ 段的規則（2026-10-05 跟使用者定案，細節手冊 §24.2）**：
- Dice 數值一律寫「起點 → 配準後」，兩個一樣大，不能只放大配準後（頭頂殘留多的人起點略高、後腦杓殘留多的 3 位起點反而低，
  只看配準後兩個方向都會被騙）。「模型貢獻」一律叫「Dice 進步多少」。
  （10-07 改正式用語後寫成「Dice（affine → 配準後）」與 ΔDice，規則不變）
- 殘留量用原始單位：頭頂、後腦杓是厚度（mm），顱底是最大一坨的體積（mm³）。相關係數、分組（170 人的 1/4、3/4 分位數）
  都直接用原始數字算（`gather.py` 的 `residue_mm`），跟以前「每批各自排名次」的版本幾乎一樣
- `1014_dilution.png`、`1014_regions.png`：橫軸殘留量、縱軸 Dice 進步多少，同一張圖裡每格縱軸刻度一樣（只是起點不同）
- `1014_six.png`（頭頂）、`1014_six_back.png`（後腦杓）、`1014_six_base.png`（顱底）同一個函式畫：test 殘留最多／最少各 3 位，每位一個切面。
  顱底的切面是穿過最大一坨殘留中心的矢狀面（每人位置不同），大字前面寫「旁邊結構」（顱底旁邊有腦幹，不能叫皮質）

📌 **第 12～15 頁「頭頂殘留厚度怎麼量」（2026-10-06，使用者：「一定要放公式、希望可以和 paper 一樣好閱讀」）**：
- 每一步一頁：上面是圖、下面是編號公式＋「其中」符號說明。公式用 LaTeX 字型（mathtext 的 Computer Modern）畫成圖，編號 (1)～(5) 靠右
- 圖和公式區塊照投影片上的實際大小畫（寬 12.13 吋），字不會被縮小
- 符號沿用使用者看懂的寫法：$z_{top}$、$\tau$、$t$、$V$、residue；$\#$＝數有幾個、$|V|$＝V 裡有幾根（解釋的過程見手冊 §24.2）
- 例子：sub-0043（test 頭頂殘留最多），第 87 片，吸管 A（x=74，乾淨，$t=0$）、B（x=101，$t=6$，中間隔一格暗的）。
  10-07 起投影片上「吸管」寫成 column、「格子」寫成 voxel
  兩根每一格都離門檻至少 0.04，不會卡在邊緣看糊塗
- 第 2 頁多一行「門檻改成皮質的 0.3～0.6 倍結論都一樣」、第 4 頁多一行「25 mm 是設計時定的範圍」：
  `make_method.py` 會重算兩種敏感度（170 人，`models/skullstrip_check/top_tau_sensitivity.csv`、`top_band_sensitivity.csv`），
  所以這一步要讀 170 個 npz，約 1～2 分鐘。門檻維持 0.5 倍是使用者 10-06 決定的（手冊 §24.2）

頁碼由 `build.js` 開頭的 `ORDER` 決定，內文引用的頁碼（第 2 頁、最後一頁）會跟著算；做出來的頁數跟 `ORDER` 對不上會直接報錯。

## 檔案

| 檔案 | 用途 |
|---|---|
| `gather.py` | 從 `models/*/dice_*.csv`、`models/folding_check/`、`models/skullstrip_check/`、第 0 步的 `models/<exp>/multipass_*.csv` 算出 `deck_data.json`；參數量用 `ASD/arch.py` 蓋網路數 |
| `make_charts.py` | 產生 `models/deck_charts/1014_*.png`（第 12～15 頁那 8 張、⑥ 的架構圖與公式區塊除外）|
| `make_method.py` | 第 12～15 頁「頭頂殘留厚度怎麼量」的 4 張圖＋4 個公式區塊，順便寫 `models/skullstrip_check/top_band_sensitivity.csv` |
| `make_arch.py` | ⑥ 第 33、34 頁的架構圖（cascade、coarse-to-fine）＋第 32～34 頁的公式區塊 (1)～(5)，不讀資料、幾秒 |
| `make_metrics.py` | 第 29 頁補充評估指標之示意圖（合成之 2D 例子）＋公式區塊 (1)～(3)，不讀資料、幾秒 |
| `build.js` | pptxgenjs 產生器：版面、文字、表格、圖片 |
| `render.ps1` | 用 PowerPoint 把每頁轉成 PNG，做版面檢查。PowerPoint 本來就開著的話不會把它關掉 |

## 重建

先算補充評估指標（HD95、SDlogJ、folding（分母：atlas 非背景）；模型或資料沒變就不用重算）。每顆約 3.5 分鐘（HD95 在 CPU 上算）；
2× width 兩顆在 8 GB 筆電 GPU 上會溢位（半精度 `--amp` 每位約 40 秒），改用 CPU 單精度（`--gpu -1`，每位約 20 秒，數值精確）。
半精度的影響可以忽略（mix_exp6 兩種精度比：SDlogJ 0.4922 vs 0.4923、HD95 逐人最大差 0.006 mm）：

```powershell
cd C:\Users\h4524\claude_cheng
.\vxm_env\Scripts\python.exe ASD\test_dice.py --baseline --test-dir data\mixed_preprocessed_v2\test --exp-name mix_exp2 --surface
foreach ($s in 'mix_exp2\0240','mix_exp5\0150','mix_exp6\0190','mix_exp7\0250','mix_exp4\0230','mix_exp3\0240') { .\vxm_env\Scripts\python.exe ASD\test_dice.py --model models\$s.pt --test-dir data\mixed_preprocessed_v2\test --surface }
foreach ($s in 'mix_wide\0225','mix_wide_vel\0240') { .\vxm_env\Scripts\python.exe ASD\test_dice.py --model models\$s.pt --test-dir data\mixed_preprocessed_v2\test --surface --gpu -1 }
```

```powershell
cd ASD\slides_src\2026-10-14_redpen
..\..\..\vxm_env\Scripts\python.exe gather.py          # -> deck_data.json
..\..\..\vxm_env\Scripts\python.exe make_charts.py     # -> models\deck_charts\1014_*.png（要先有 deck_data.json）
..\..\..\vxm_env\Scripts\python.exe make_method.py     # -> 1014_method_{1..4}.png、1014_method_eq{1..4}.png（約 1～2 分鐘）
..\..\..\vxm_env\Scripts\python.exe make_arch.py       # -> 1014_arch_{cascade,pyramid}.png、1014_arch_{step0,cascade,pyramid}_eq.png
cd ..\..\..
.\vxm_env\Scripts\python.exe ASD\check_folding.py --models mix_exp6:0190 mix_exp7:0250 mix_wide_vel:0240 --gpu 0   # 速度場三顆的擠爆位置（位移場三顆 09-30 算過）
.\vxm_env\Scripts\python.exe ASD\check_folding.py --plot-only --views   # -> folding_views.png、folding_params.png（6 欄：左 3 顆速度場、右 3 顆位移場）
.\vxm_env\Scripts\python.exe ASD\check_folding.py --plot-only --gpu 0   # -> folding_zoom_T054.png（第 6 頁；mix_exp3 推論一次，筆電 GPU 可）
cd ASD\slides_src\2026-09-20_cross
..\..\..\vxm_env\Scripts\python.exe make_compare.py --set lambda   # -> curve_lambda / jacobian_lambda / grid_lambda.png（要先有各顆的 vis_T054）
..\..\..\vxm_env\Scripts\python.exe make_compare.py --set wide     # -> curve_wide / jacobian_wide / grid_wide.png（簡報只用 curve_wide）
cd ..\..\..
.\vxm_env\Scripts\python.exe ASD\check_folding.py --zoom-pair --gpu 0   # -> folding_check\folding_full_pair_T054.png、folding_zoom_pair_T054.png（第 27、28 頁）
cd ASD\slides_src\2026-09-20_cross
cd ..\2026-10-14_redpen
node build.js deck.pptx
powershell -ExecutionPolicy Bypass -File render.ps1 -Pptx deck.pptx -OutDir render
copy deck.pptx ..\..\..\meeting報告\ASD_老師紅字回覆_20261014.pptx
```

🔴 **複製到 `meeting報告\` 之前，先確認那份沒被使用者改過**（`fc /b` 或比修改時間），**也沒開著**。
使用者會在 PowerPoint 裡加字，覆蓋掉就沒了（09-30 那份就是這樣）。
⚠️ 看有沒有開著要找鎖檔 `meeting報告\~$ASD_老師紅字回覆_20261014.pptx`（例如 `ls meeting報告/ | grep -F '~$'`）。
2026-10-06 發現之前用 `grep "~\$"` 檢查，在 bash 雙引號裡變成「行尾是 ~」，永遠找不到鎖檔；
所幸 PowerPoint 開著時 Windows 不讓覆蓋（`cp` 會報 Device or resource busy），所以沒有蓋掉過使用者的修改

`build.js` 會去 `..\2026-09_mix_tigerbx\node_modules` 找 pptxgenjs（跟 09-20 那份一樣）。

## 還沒跑完的實驗

mix_exp6、mix_exp7（10-04）、mix_wide_vel（10-05）都帶回來了，現在沒有 pending 的。
`gather.py` 找不到某顆的 test CSV（`models/<exp>/dice_<4 位數>.csv`）就標成 pending，簡報第 2 頁、平滑權重那頁、加寬＋速度場那頁顯示「訓練中」、圖上標「訓練中」框。
**結果帶回來放進 `models/<exp>/` 之後，照上面重建一次就會補上**，不用改程式。
一個資料夾有好幾份 test CSV 的話，用 `dice_curve_val.csv` 裡驗證集最好的那個 epoch。
mix_exp8／9（訓練時也用標籤）不在這份簡報的設定裡，只在「下一步」寫一行；結果要放進來得另外加頁。
⑥ 架構修改的三顆（mix_cascade、mix_pyramid、mix_cascade_pyramid）在 `gather.py` 的 `CFG` 裡，還沒跑 → 第 35 頁 2 × 2 表寫「待訓練」（mix_cascade_pyramid 寫「待前兩者結果」）；
結果帶回來（`models/<exp>/dice_<epoch>.csv`）重建就會換成 Dice。

## 寫死在程式裡、不是從 CSV 來的

- 每顆的設定（版本、實際平滑權重、寬度、積分解析度）：`gather.py` 的 `CFG`，照操作單
- 論文 Table I 的擠爆比例 0.366%（`make_charts.py`）
- 對照圖的兩位受試者 MRS0381-2、T054（跟 `check_top_residue.py --example` 同一組）
- 第 2 頁結果欄的結論句（「呈小型團簇，沿腦溝分布於皮質與白質」「有殘留，與 ΔDice 無顯著相關」等）、各頁標題的結論句，是照現在的結果寫的
- 第 16 頁 2×2 表最後一列「affine 已高」「ΔDice 相近」「affine 相近」、第 19 頁「兩組之 ΔDice 互有高低，無法區分」、
  第 20 頁「此 6 位中，殘留多者之 ΔDice 略高」「殘留多位於腦之前下方」這幾個字是照現在的數字寫的（數字本身從 CSV 算）
- 第 12～15 頁（`make_method.py`）：例子 sub-0043、第 87 片、column x=74／101；「腦膜與腦組織之間原有一層腦脊髓液」「該處頭皮傾斜」
  這些解釋是寫死的文字；圖上和公式說明裡的數字（0.62、0.31、0.23～0.32、8,980、31,524、3.51、r 範圍）都從資料算
- 第 2 頁的「註：SVF = stationary velocity field…」（2026-10-07 使用者：「P2 就和老師說 SVF」）
- 第 30 頁（補充評估指標結果）的標題與三行重點：數字從 `gather.py` 的 `surface` 讀，句子是照 2026-10-07 的結果寫的
  （HD95 各模型相近、SDlogJ 隨 λ 變小而上升、displacement field 之 SDlogJ 主要來自 folding voxel、λ = 1 之 folding 高於論文）
- 第 5、9、24 頁圖上的英文結構名稱：第 9、24 頁是 FreeSurferColorLUT 名稱左右合併（`gather.py` 的 `struct_en`）；
  第 5 頁是區域分組（多個標籤），英文是描述性的（`make_charts.py` 的 `SHOW`）
- 第 22 頁原本的附記（約 33 GB／24 GB、每步秒數）已拿掉；`gather.py` 仍算 `train_time`（從 `log/mix_wide*.txt`），簡報不再使用
- 第 24 頁「蒼白球…方向相反」、第 26 頁「平滑項不可跨參數化比較」、第 27、28 頁的說明文字是照現在的數字寫的
- 第 37 頁「半監督訓練…γ = 0.5、5」：照 `ASD/指令_mix_exp8_9.md`；「Eq. 10」照 `ASD/train_semisup.py` 檔頭（論文式 (9)、(10)）
- ⑥ 架構修改（第 31～35 頁）：文獻數字照論文抄、寫在 `make_charts.py`（Jian et al., WBIR 2024 Table 2 的 LPBA 欄
  67.0／67.5／67.3／70.4／71.3，2026-10-06 對過原文 HTML；LUMIR 2024 測試集 Dice，出處見 `文獻/對位模型文獻筆記.md`），
  第 31 頁的資料來源與四行重點寫在 `build.js`；第 32 頁右側重點的數字從 CSV 算，文字（「3 passes 無進一步改善」等）照現在的數字寫；
  第 32～34 頁的公式 (1)～(5) 與「其中」說明（λ = 1、7 steps、參數 0.41 M、引用文獻）寫在 `make_arch.py`；
  第 35 頁每顆的每步倍數、顯存、訓練時間寫在 `gather.py` 的 `arch`（筆電實測＋外插，手冊 §25.3、§25.4），參數是由網路計算的；
  「實作驗證」「訓練順序」兩行是照現在的狀態寫的文字

## 不進版控

`deck_data.json`、`deck.pptx`、`render/`
