param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = ('itemcf_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [int]$BenchmarkUsers = 256
)
$ErrorActionPreference = 'Stop'
$itemcfRepo = Split-Path -Parent $PSScriptRoot
$itemcfPython = Join-Path $itemcfRepo '.venv/Scripts/python.exe'
function Invoke-ItemCFPython {
    param([string[]]$PythonArguments)
    & $itemcfPython @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "ItemCF step failed: $PythonArguments" }
}
Push-Location $itemcfRepo
try {
    $itemcfRoot = "artifacts/$RunName"
    if ((Test-Path -LiteralPath $itemcfRoot) -or (Test-Path -LiteralPath "experiments/$RunName")) { throw 'Choose a new RunName; existing experiments are preserved.' }
    Invoke-ItemCFPython -PythonArguments @('-m', 'src.itemcf_index', 'build', '--config', 'configs/itemcf_index_cosine.json', '--out', "$itemcfRoot/cap200")
    Invoke-ItemCFPython -PythonArguments @('-m', 'src.itemcf_index', 'build', '--config', 'configs/itemcf_index_full_cosine.json', '--out', "$itemcfRoot/full_cosine")
    Invoke-ItemCFPython -PythonArguments @('-m', 'src.itemcf_index', 'build', '--config', 'configs/itemcf_index_full_pop025.json', '--out', "$itemcfRoot/full_pop025")
    $itemcfSpecs = Get-Content -LiteralPath 'configs/itemcf_ablation_full.json' -Raw | ConvertFrom-Json
    $itemcfSpecs.indexes.cap200_cosine = "$itemcfRoot/cap200"
    $itemcfSpecs.indexes.full_cosine = "$itemcfRoot/full_cosine"
    $itemcfSpecs.indexes.full_pop025 = "$itemcfRoot/full_pop025"
    [System.IO.File]::WriteAllText((Join-Path $itemcfRepo "$itemcfRoot/ablations.json"), ($itemcfSpecs | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
    Invoke-ItemCFPython -PythonArguments @('-m', 'src.itemcf_evaluate', '--variants', "$itemcfRoot/ablations.json", '--out', "experiments/$RunName", '--benchmark-users', "$BenchmarkUsers", '--repeats', '3')
    Invoke-ItemCFPython -PythonArguments @('-m', 'src.itemcf_index', 'export', '--index', "$itemcfRoot/full_pop025", '--out', "$itemcfRoot/balanced", '--neighbors', '300')
    $itemcfCascade = Get-Content -LiteralPath 'configs/cascade_v2_itemcf_index.json' -Raw | ConvertFrom-Json
    $itemcfCascade.itemcf_index = "$itemcfRoot/balanced"
    [System.IO.File]::WriteAllText((Join-Path $itemcfRepo "$itemcfRoot/cascade_balanced.json"), ($itemcfCascade | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
} finally {
    Pop-Location
}
