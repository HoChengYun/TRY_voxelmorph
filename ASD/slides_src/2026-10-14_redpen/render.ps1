param(
    [Parameter(Mandatory=$true)][string]$Pptx,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [int]$Width = 1600,
    [int]$Height = 900
)
# 用使用者自己的 PowerPoint 把每一頁轉成 PNG，做版面檢查（真正的 PowerPoint 算字寬，中文最可靠）。
# 跟 2026-09-20_cross 那支一樣，差別：PowerPoint 本來就開著的話不呼叫 Quit()，免得關掉使用者正在編輯的簡報。
$ErrorActionPreference = 'Stop'
$Pptx = (Resolve-Path $Pptx).Path
if (Test-Path $OutDir) { Get-ChildItem $OutDir -Filter 'slide-*.png' | Remove-Item -Force }
else { New-Item -ItemType Directory -Force $OutDir | Out-Null }
$OutDir = (Resolve-Path $OutDir).Path

$wasRunning = [bool](Get-Process POWERPNT -ErrorAction SilentlyContinue)
$app = New-Object -ComObject PowerPoint.Application
try {
    # ReadOnly=1, Untitled=0, WithWindow=0
    $pres = $app.Presentations.Open($Pptx, 1, 0, 0)
    try {
        $n = $pres.Slides.Count
        for ($i = 1; $i -le $n; $i++) {
            $name = 'slide-{0:D2}.png' -f $i
            $pres.Slides.Item($i).Export((Join-Path $OutDir $name), 'PNG', $Width, $Height)
        }
        Write-Output ("exported {0} slides -> {1}" -f $n, $OutDir)
    } finally { $pres.Close() }
} finally {
    if (-not $wasRunning) { $app.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
