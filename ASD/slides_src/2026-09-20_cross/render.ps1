param(
    [Parameter(Mandatory=$true)][string]$Pptx,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [int]$Width = 1600,
    [int]$Height = 900
)
# 用使用者自己的 PowerPoint 把每一頁轉成 PNG，做版面檢查。
# 用真正的 PowerPoint 算字寬，比 LibreOffice 替代字型可靠（中文字型尤其如此）。
$ErrorActionPreference = 'Stop'
$Pptx = (Resolve-Path $Pptx).Path
if (Test-Path $OutDir) { Get-ChildItem $OutDir -Filter 'slide-*.png' | Remove-Item -Force }
else { New-Item -ItemType Directory -Force $OutDir | Out-Null }
$OutDir = (Resolve-Path $OutDir).Path

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
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
