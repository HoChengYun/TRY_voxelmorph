# =====================================================================
#  ASD 訓練：VoxelMorph scan-to-atlas
# =====================================================================
#
#  用法（在專案根目錄執行，路徑會自動偵測，換機器不用改）：
#
#      .\ASD\run_train.ps1                       # 預設 ncc + lambda 1.0, 250 epochs
#      .\ASD\run_train.ps1 -ExpName asd_exp2 -ImageLoss mse -Lambda 0.01
#      .\ASD\run_train.ps1 -Epochs 100           # 先跑短的看看
#      .\ASD\run_train.ps1 -Resume               # 從已存的最後一個 epoch 續跑
#      .\ASD\run_train.ps1 -CheckOnly            # 只做起跑前檢查，不訓練
#
#  中斷了用 -Resume 續跑，不用從頭來。
#
#  🔴 lambda 的尺度取決於 image-loss：
#       ncc -> 論文最佳約 1~2   （train.py 預設的 0.01 對 NCC 等於幾乎沒有正則化）
#       mse -> 論文最佳約 0.01~0.02
# =====================================================================

param(
    [string]$ExpName   = 'asd_exp1',
    [string]$ImageLoss = 'ncc',
    [double]$Lambda    = 1.0,
    [int]$Epochs       = 250,
    [int]$StepsPerEpoch = 100,
    [string]$Gpu       = '0',
    [switch]$Resume,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'

# 由腳本位置自動推出專案根目錄，換機器 clone 到別的路徑也不用改
$Root     = Split-Path -Parent $PSScriptRoot
$Python   = Join-Path $Root 'vxm_env\Scripts\python.exe'
$Train    = Join-Path $Root 'voxelmorph-code\scripts\torch\train.py'
$DataDir  = Join-Path $Root 'ASD\ASD_preprocessed_v1\train'
$TestDir  = Join-Path $Root 'ASD\ASD_preprocessed_v1\test'
$Atlas    = Join-Path $Root 'IXI\atlas_mni152_09c_v3.npz'
$ModelDir = Join-Path $Root ('models\' + $ExpName)
$LogDir   = Join-Path $Root 'log'
$LogFile  = Join-Path $LogDir ($ExpName + '.txt')
$CmdFile  = Join-Path $LogDir ($ExpName + '_script.txt')

Set-Location $Root
$env:PYTHONIOENCODING = 'utf-8'

Write-Host ''
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ('  ASD 訓練：' + $ExpName) -ForegroundColor Cyan
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ''

# ── 起跑前檢查 ───────────────────────────────────────────────────────
Write-Host '[1/3] 起跑前檢查' -ForegroundColor Yellow
Write-Host ('      專案根目錄：' + $Root)

$ok = $true
foreach ($p in @($Python, $Train, $Atlas)) {
    if (Test-Path $p) { Write-Host ('      [v] ' + $p) }
    else { Write-Host ('      [X] 找不到：' + $p) -ForegroundColor Red; $ok = $false }
}

foreach ($d in @($DataDir, $TestDir)) {
    if (Test-Path $d) {
        $n = (Get-ChildItem $d -Filter *.npz -ErrorAction SilentlyContinue).Count
        Write-Host ('      [v] ' + $d + '  (' + $n + ' 個 npz)')
        if ($n -eq 0) { Write-Host '      [X] 資料夾是空的' -ForegroundColor Red; $ok = $false }
    } else {
        Write-Host ('      [X] 找不到：' + $d) -ForegroundColor Red
        Write-Host '          前處理過的 npz 需要從另一台機器搬過來（約 1.26 GB），' -ForegroundColor Red
        Write-Host '          它不在 git 裡（.gitignore 有擋）。' -ForegroundColor Red
        $ok = $false
    }
}

# GPU
$gpuInfo = & $Python -c "import torch; print('OK' if torch.cuda.is_available() else 'NO'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('%.1f' % (torch.cuda.get_device_properties(0).total_memory/1024**3) if torch.cuda.is_available() else '0')"
$gpuLines = $gpuInfo -split "`n"
if ($gpuLines[0].Trim() -eq 'OK') {
    Write-Host ('      [v] GPU: ' + $gpuLines[1].Trim() + '  (' + $gpuLines[2].Trim() + ' GB VRAM)')
    if ([double]$gpuLines[2].Trim() -lt 7.5) {
        Write-Host '      [!] VRAM 小於 7.5 GB，192x224x192 可能會 OOM' -ForegroundColor Yellow
    }
} else {
    Write-Host '      [X] 沒有偵測到 CUDA GPU' -ForegroundColor Red
    Write-Host '          NCC loss 的 sum_filt 寫死在 cuda 上（losses.py:28），沒 GPU 跑不了' -ForegroundColor Red
    $ok = $false
}

# atlas 版本要跟資料版本一致
Write-Host ('      [i] atlas: ' + (Split-Path $Atlas -Leaf) + '  <- 必須跟前處理用的同一版')

if (-not $ok) {
    Write-Host ''
    Write-Host '  起跑前檢查未通過，已中止。' -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory $LogDir | Out-Null }

# ── 續跑判斷 ─────────────────────────────────────────────────────────
$InitialEpoch = 0
if (Test-Path $ModelDir) {
    $pts = Get-ChildItem $ModelDir -Filter '*.pt' -ErrorAction SilentlyContinue | Sort-Object Name
    if ($pts.Count -gt 0) {
        $last = [int]($pts[-1].BaseName)
        if ($Resume) {
            $InitialEpoch = $last
            Write-Host ('      [i] -Resume：從 epoch ' + $last + ' 續跑')
        } else {
            Write-Host ''
            Write-Host ('  [!] ' + $ModelDir + ' 已經有 ' + $pts.Count + ' 個 .pt（最後是 ' + $pts[-1].Name + '）') -ForegroundColor Yellow
            Write-Host '      直接跑會從 epoch 0 覆蓋。要續跑請加 -Resume，或換一個 -ExpName。' -ForegroundColor Yellow
            $ans = Read-Host '      仍要從頭開始？(y/N)'
            if ($ans -ne 'y' -and $ans -ne 'Y') { Write-Host '  已取消。'; exit 0 }
        }
    }
}

# ── 參數摘要 ─────────────────────────────────────────────────────────
Write-Host ''
Write-Host '[2/3] 訓練設定' -ForegroundColor Yellow
Write-Host ('      實驗名稱     : ' + $ExpName)
Write-Host ('      image-loss   : ' + $ImageLoss)
Write-Host ('      lambda       : ' + $Lambda + '  <- ncc 建議 1~2；mse 建議 0.01~0.02')
Write-Host ('      epochs       : ' + $Epochs + ' (從 ' + $InitialEpoch + ' 開始)')
Write-Host ('      steps/epoch  : ' + $StepsPerEpoch)
Write-Host ('      訓練資料     : ' + $DataDir)
Write-Host ('      模型輸出     : ' + $ModelDir)
Write-Host ('      記錄檔       : ' + $LogFile)
Write-Host ''
Write-Host ('      預估：每步約 1.5 秒 -> ' + [math]::Round(($Epochs - $InitialEpoch) * $StepsPerEpoch * 1.5 / 3600, 1) + ' 小時（實際看 GPU）')
Write-Host ('      磁碟：每個 epoch 存一個 .pt（1.16 MB）-> 約 ' + [math]::Round($Epochs * 1.16, 0) + ' MB')

if ($CheckOnly) {
    Write-Host ''
    Write-Host '  -CheckOnly 模式，到此結束。' -ForegroundColor Cyan
    exit 0
}

# ── 組指令並記錄 ─────────────────────────────────────────────────────
$TrainArgs = @(
    $Train, $DataDir,
    '--atlas',           $Atlas,
    '--model-dir',       $ModelDir,
    '--epochs',          $Epochs,
    '--steps-per-epoch', $StepsPerEpoch,
    '--gpu',             $Gpu,
    '--image-loss',      $ImageLoss,
    '--lambda',          $Lambda
)
if ($InitialEpoch -gt 0) {
    $TrainArgs += @('--initial-epoch', $InitialEpoch,
                    '--load-model', (Join-Path $ModelDir ('{0:d4}.pt' -f $InitialEpoch)))
}

# 專案慣例：每次跑都要留指令記錄，之後才查得出當初的參數
$cmdText = @"
# $ExpName 訓練指令記錄
# 執行時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
# 機器: $env:COMPUTERNAME
# 資料: ASD_preprocessed_v1 (train 150 / test 17, grouping=none, 167 subjects)
# 由 ASD\run_train.ps1 產生

$Python $($TrainArgs -join ' ')
"@
$cmdText | Out-File -FilePath $CmdFile -Encoding utf8

Write-Host ''
Write-Host '[3/3] 開始訓練' -ForegroundColor Yellow
Write-Host ('      指令已記錄到 ' + $CmdFile)
Write-Host ('      另開視窗看進度：Get-Content ' + $LogFile + ' -Tail 20 -Wait')
Write-Host ''

$t0 = Get-Date
& $Python $TrainArgs 2>&1 | Tee-Object -FilePath $LogFile
$exit = $LASTEXITCODE
$hrs = [math]::Round(((Get-Date) - $t0).TotalHours, 2)

Write-Host ''
Write-Host ('      耗時 ' + $hrs + ' 小時')

if ($exit -ne 0) {
    Write-Host ('  訓練回傳非零結束碼 (' + $exit + ')，請看 ' + $LogFile) -ForegroundColor Red
    Write-Host '  中斷的話可以用 -Resume 從最後一個 .pt 續跑。' -ForegroundColor Yellow
    exit $exit
}

$nPt = (Get-ChildItem $ModelDir -Filter '*.pt').Count
Write-Host ('      產出 ' + $nPt + ' 個 .pt')

Write-Host ''
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host '  訓練完成 — 下一步：挑最佳 epoch' -ForegroundColor Cyan
Write-Host '=====================================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host ('    ' + $Python + ' voxelmorph-code\scripts\torch\batch_test_ixi.py `')
Write-Host ('        --model-dir ' + $ModelDir + ' `')
Write-Host ('        --atlas ' + $Atlas + ' `')
Write-Host ('        --test-dir ' + $TestDir + ' `')
Write-Host ('        --out-dir ' + $ModelDir + ' --step 1 --gpu ' + $Gpu)
Write-Host ''
Write-Host '  會產生 epoch_curve.csv / .png，用來挑 epoch。'
Write-Host '  ⚠️ 目前沒有 val set，epoch 是依 test 曲線挑的 —— 發表時方法學要明講。'
Write-Host ''
Write-Host '  要搬回主機的只有：最佳的幾個 .pt、epoch_curve.csv/.png、log。'
Write-Host ('  不用把 ' + $nPt + ' 個 .pt 全搬。')
Write-Host ''
