# =====================================================================
#  ASD 資料前處理：FreeSurfer 產物 -> VoxelMorph 訓練用 npz
#  =====================================================================
#
#  用法（在 C:\Users\h4524\claude_cheng 底下執行）：
#
#      .\ASD\run_preprocess.ps1              # 正常跑（會先顯示切分並要你確認）
#      .\ASD\run_preprocess.ps1 -DryRun      # 只看切分，不動影像
#      .\ASD\run_preprocess.ps1 -Force       # 跳過確認，直接開跑
#      .\ASD\run_preprocess.ps1 -SaveNii     # 額外輸出每顆的 .nii.gz（多佔約 1.4 GB）
#
#  中斷了可以直接重跑 —— 腳本預設 --skip-done，已完成的會略過。
#
#  預計耗時：167 顆 x 約 25 秒 = 70~90 分鐘（主要花在 ANTs Affine 配準）
# =====================================================================

param(
    [switch]$DryRun,
    [switch]$Force,
    [switch]$SaveNii
)

$ErrorActionPreference = 'Stop'

# ── 路徑設定 ─────────────────────────────────────────────────────────
# 由腳本位置自動推出專案根目錄（ASD\run_preprocess.ps1 -> 上一層的上一層），
# 換一台機器 clone 到別的路徑也不用改。
$Root     = Split-Path -Parent $PSScriptRoot
$Python   = Join-Path $Root 'vxm_env\Scripts\python.exe'
$Script   = Join-Path $Root 'ASD\preprocess_fs.py'
$ImgDir   = Join-Path $Root 'ASD\ASD_data\norm'
$SegDir   = Join-Path $Root 'ASD\ASD_data\aseg'
$Atlas    = Join-Path $Root 'IXI\atlas_mni152_09c_v3.nii.gz'
$OutDir   = Join-Path $Root 'ASD\ASD_preprocessed_v1'
$SubjList = Join-Path $Root 'ASD\subjects_final.txt'
$LogDir   = Join-Path $Root 'log'
$LogFile  = Join-Path $LogDir 'asd_preprocess.txt'
$CmdFile  = Join-Path $LogDir 'asd_preprocess_script.txt'

Set-Location $Root
$env:PYTHONIOENCODING = 'utf-8'

Write-Host ''
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host '  ASD 前處理' -ForegroundColor Cyan
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ''

# ── 起跑前檢查 ───────────────────────────────────────────────────────
Write-Host '[1/4] 起跑前檢查' -ForegroundColor Yellow

$ok = $true
foreach ($p in @($Python, $Script, $Atlas, $SubjList)) {
    if (Test-Path $p) {
        Write-Host ("      [v] " + $p)
    } else {
        Write-Host ("      [X] 找不到：" + $p) -ForegroundColor Red
        $ok = $false
    }
}

foreach ($d in @($ImgDir, $SegDir)) {
    if (Test-Path $d) {
        $n = (Get-ChildItem $d -Filter *.nii.gz -ErrorAction SilentlyContinue).Count
        $mb = [math]::Round((Get-ChildItem $d -Filter *.nii.gz | Measure-Object Length -Sum).Sum / 1MB, 0)
        Write-Host ("      [v] " + $d + "  (" + $n + " 個, " + $mb + " MB)")
        if ($n -ne 167) {
            Write-Host ("      [!] 預期 167 個，實際 " + $n + " 個") -ForegroundColor Yellow
        }
    } else {
        Write-Host ("      [X] 找不到：" + $d) -ForegroundColor Red
        $ok = $false
    }
}

# 清單裡的 ID 數
$nSubj = (Get-Content $SubjList | Where-Object { $_.Trim() -ne '' -and -not $_.Trim().StartsWith('#') }).Count
Write-Host ("      [v] 受試者清單：" + $nSubj + " 個 ID")

# 資料不該進 git
$probe = Join-Path $ImgDir ((Get-ChildItem $ImgDir -Filter *.nii.gz | Select-Object -First 1).Name)
$null = & git check-ignore -q $probe 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host '      [v] .gitignore 有擋住影像資料'
} else {
    Write-Host '      [!] 警告：影像資料沒有被 .gitignore 擋住，不要 commit！' -ForegroundColor Red
}

if (-not $ok) {
    Write-Host ''
    Write-Host '  起跑前檢查未通過，已中止。' -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory $LogDir | Out-Null }

# ── 共用參數 ─────────────────────────────────────────────────────────
#   --grouping none  = 每個掃描各自成一人（使用者 2026-08-23 的決定）
#                      A013/A0131/A0132 與 A016_1/A016_2 都當不同人
#   --list-is-final  = 混掃描檢查已通過，解除批次閘門
$ArgsCommon = @(
    $Script,
    '--img-dir',      $ImgDir,
    '--seg-dir',      $SegDir,
    '--atlas',        $Atlas,
    '--out-dir',      $OutDir,
    '--subject-list', $SubjList,
    '--grouping',     'none'
)

# ── 先跑 dry-run 顯示切分 ────────────────────────────────────────────
Write-Host ''
Write-Host '[2/4] 切分預覽（不動影像）' -ForegroundColor Yellow
Write-Host ''
& $Python ($ArgsCommon + '--dry-run')
if ($LASTEXITCODE -ne 0) {
    Write-Host '  dry-run 失敗，已中止。' -ForegroundColor Red
    exit 1
}

if ($DryRun) {
    Write-Host ''
    Write-Host '  -DryRun 模式，到此結束。' -ForegroundColor Cyan
    exit 0
}

# ── 確認 ─────────────────────────────────────────────────────────────
if (-not $Force) {
    Write-Host ''
    Write-Host '---------------------------------------------------------------------'
    Write-Host ('  即將處理 ' + $nSubj + ' 顆，預計 70~90 分鐘。')
    Write-Host ('  輸出到：' + $OutDir)
    Write-Host ('  記錄檔：' + $LogFile)
    Write-Host '  （中斷後直接重跑即可續跑，已完成的會略過）'
    Write-Host '---------------------------------------------------------------------'
    $ans = Read-Host '  確定開始？(y/N)'
    if ($ans -ne 'y' -and $ans -ne 'Y') {
        Write-Host '  已取消。'
        exit 0
    }
}

# ── 正式跑 ───────────────────────────────────────────────────────────
$ArgsRun = $ArgsCommon + '--list-is-final'
if ($SaveNii) { $ArgsRun = $ArgsRun + '--save-nii' }

# 把指令存檔（專案慣例：每次跑都要留指令記錄）
$cmdText = @"
# ASD 前處理指令記錄
# 執行時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
# 由 ASD\run_preprocess.ps1 產生

$Python $($ArgsRun -join ' ')
"@
$cmdText | Out-File -FilePath $CmdFile -Encoding utf8

Write-Host ''
Write-Host '[3/4] 開始批次處理' -ForegroundColor Yellow
Write-Host ('      指令已記錄到 ' + $CmdFile)
Write-Host ('      進度可另開視窗看：Get-Content ' + $LogFile + ' -Tail 20 -Wait')
Write-Host ''

$t0 = Get-Date
& $Python $ArgsRun 2>&1 | Tee-Object -FilePath $LogFile
$exit = $LASTEXITCODE
$mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

Write-Host ''
Write-Host ('      耗時 ' + $mins + ' 分鐘')

if ($exit -ne 0) {
    Write-Host ''
    Write-Host ('  前處理回傳非零結束碼 (' + $exit + ')，請看 ' + $LogFile) -ForegroundColor Red
    exit $exit
}

# ── 跑完檢查 ─────────────────────────────────────────────────────────
Write-Host ''
Write-Host '[4/4] 跑完檢查' -ForegroundColor Yellow

$nTrain = (Get-ChildItem (Join-Path $OutDir 'train') -Filter *.npz -ErrorAction SilentlyContinue).Count
$nTest  = (Get-ChildItem (Join-Path $OutDir 'test')  -Filter *.npz -ErrorAction SilentlyContinue).Count
$total  = $nTrain + $nTest

Write-Host ('      train : ' + $nTrain + ' 個 npz')
Write-Host ('      test  : ' + $nTest  + ' 個 npz')
Write-Host ('      合計  : ' + $total  + ' / ' + $nSubj)

if ($total -ne $nSubj) {
    Write-Host ('      [!] 數量對不上，預期 ' + $nSubj + '，請查 ' + $LogFile + ' 裡的失敗記錄') -ForegroundColor Yellow
} else {
    Write-Host '      [v] 數量正確' -ForegroundColor Green
}

# 抽驗 3 顆做量化檢查（含左右翻轉檢查）
Write-Host ''
Write-Host '      抽驗 3 顆：'
$verify = Join-Path $Root 'ASD\verify_one_subject.py'
if (Test-Path $verify) {
    $samples = Get-ChildItem (Join-Path $OutDir 'train') -Filter *.npz | Get-Random -Count ([Math]::Min(3, $nTrain))
    foreach ($f in $samples) {
        $out = & $Python $verify --npz $f.FullName 2>&1
        $line = $out | Select-String '結果：' | Select-Object -First 1
        if ($LASTEXITCODE -eq 0) {
            Write-Host ('        [v] ' + $f.BaseName + '  ' + $line) -ForegroundColor Green
        } else {
            Write-Host ('        [X] ' + $f.BaseName + '  ' + $line) -ForegroundColor Red
            Write-Host '            完整輸出：'
            $out | ForEach-Object { Write-Host ('            ' + $_) }
        }
    }
}

Write-Host ''
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host '  完成' -ForegroundColor Cyan
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host '  下一步（訓練）：'
Write-Host ''
Write-Host ('    ' + $Python + ' voxelmorph-code\scripts\torch\train.py ' + $OutDir + '\train `')
Write-Host '        --atlas IXI\atlas_mni152_09c_v3.npz `'
Write-Host '        --model-dir models\asd_exp1 `'
Write-Host '        --epochs 250 --gpu 0 `'
Write-Host '        --image-loss ncc --lambda 1.0 > .\log\asd_exp1.txt 2>&1'
Write-Host ''
Write-Host '  注意 --image-loss 預設是 mse，跑 NCC 一定要明寫；'
Write-Host '  lambda 對 NCC 的合適尺度約 1~2（不是 train.py 預設的 0.01）。'
Write-Host ''
