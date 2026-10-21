param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = ('pairwise_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [string]$Config = 'configs/pairwise_v1.json'
)
$ErrorActionPreference = 'Stop'
$pairRoot = Split-Path -Parent $PSScriptRoot
$pairPython = Join-Path $pairRoot '.venv/Scripts/python.exe'
function Invoke-PairPython([string[]]$Arguments) {
    & $pairPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pairwise stage failed: $Arguments" }
}
Push-Location $pairRoot
try {
    $run = "artifacts/$RunName"
    if (Test-Path -LiteralPath $run) { throw 'Choose a new RunName; preserve existing outputs.' }
    if (-not (Test-Path -LiteralPath $pairPython)) { throw 'Prepare the project Python environment first.' }
    Invoke-PairPython @('-m','src.v2.pairwise_data','--config',$Config,'--out',"$run/initial/pairs")
    Invoke-PairPython @('-m','src.v2.pairwise_train','--config',$Config,'--pairs',"$run/initial/pairs",'--out',"$run/initial/models")
    $extra = Get-Content -LiteralPath 'configs/pairwise_fullmix_v1.json' -Raw | ConvertFrom-Json
    $extra.base_run = "$run/initial"
    [System.IO.File]::WriteAllText((Join-Path $pairRoot "$run/followup.json"), ($extra | ConvertTo-Json -Depth 20), [System.Text.UTF8Encoding]::new($false))
    Invoke-PairPython @('-m','src.v2.pairwise_fullmix','--config',"$run/followup.json",'--out',"$run/final/models")
    Invoke-PairPython @('-m','src.v2.pairwise_evaluate','--run',"$run/final",'--out',"$run/final/evaluation")
    Invoke-PairPython @('-m','src.v2.pairwise_audit','--run',"$run/final",'--pairs',"$run/initial/pairs",'--out',"$run/final/audit.json")
    Write-Host "Completed $run/final; default models unchanged."
} finally { Pop-Location }
