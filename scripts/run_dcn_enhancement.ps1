param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = ('dcn_enhancement_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [string]$Config = 'configs/dcn_enhancement_v1.json',
    [int]$RuntimeUsers = 256
)
$ErrorActionPreference = 'Stop'
$enhanceRoot = Split-Path -Parent $PSScriptRoot
$enhancePython = Join-Path $enhanceRoot '.venv/Scripts/python.exe'
function Invoke-EnhancedPython([string[]]$Arguments) {
    & $enhancePython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Enhanced DCN stage failed: $Arguments" }
}
function Write-EnhancedJson([string]$Path, $Value) {
    [System.IO.File]::WriteAllText((Join-Path $enhanceRoot $Path), ($Value | ConvertTo-Json -Depth 20), [System.Text.UTF8Encoding]::new($false))
}
Push-Location $enhanceRoot
try {
    $run = "artifacts/$RunName"
    if (Test-Path -LiteralPath $run) { throw 'Choose a new RunName; existing outputs are preserved.' }
    if (-not (Test-Path -LiteralPath $enhancePython)) { throw 'Run scripts/setup_v2.ps1 with Python 3.12 first.' }
    $protocol = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
    if (-not (Test-Path -LiteralPath 'artifacts/itemcf_support2_300/metadata.json')) {
        Invoke-EnhancedPython @('-m','src.itemcf_index','build','--config','configs/itemcf_index_support2.json','--out','artifacts/itemcf_support2_300')
    }
    Invoke-EnhancedPython @('-c','from pathlib import Path; import sys; from src.v2.enhanced_data import prepare; prepare(Path(sys.argv[1]), sample_rows=int(sys.argv[2]), sample_seed=int(sys.argv[3]), minimum=int(sys.argv[4]), length=int(sys.argv[5]))',
        "$run/cache","$($protocol.sample_rows)","$($protocol.sample_seed)","$($protocol.minimum_frequency)","$($protocol.history_length)")
    Invoke-EnhancedPython @('-m','src.v2.enhanced_audit','--cache',"$run/cache",'--out',"$run/audit")
    Invoke-EnhancedPython @('-m','src.v2.train_rankers','--kind','coarse_lr','--out',"$run/baseline/coarse_lr")
    Invoke-EnhancedPython @('-m','src.v2.train_rankers','--kind','fine_dcn','--out',"$run/baseline/fine_dcn")
    $recall = Get-Content -LiteralPath $protocol.recall_config -Raw | ConvertFrom-Json
    $recall.coarse_model = "$run/baseline/coarse_lr/model.pt"
    $recall.fine_model = "$run/baseline/fine_dcn/model.pt"
    Write-EnhancedJson "$run/recall.json" $recall
    Invoke-EnhancedPython @('-m','src.v2.enhanced_evaluate','freeze','--config',"$run/recall.json",'--out',"$run/candidates")
    Invoke-EnhancedPython @('-m','src.v2.enhanced_train','--config',$Config,'--cache',"$run/cache",'--out',"$run/models")
    Invoke-EnhancedPython @('-m','src.v2.enhanced_evaluate','evaluate','--cache',"$run/cache",'--models',"$run/models",'--pool',"$run/candidates",'--out',"$run/evaluation")
    $evaluation = Get-Content -LiteralPath "$run/evaluation/metrics.json" -Raw | ConvertFrom-Json
    $recall | Add-Member -NotePropertyName 'baseline_fine_model' -NotePropertyValue "$run/baseline/fine_dcn/model.pt"
    $recall | Add-Member -NotePropertyName 'enhanced_data' -NotePropertyValue "$run/cache"
    $recall.fine_model = "$run/models/$($evaluation.cascade_selected_model)/model.pt"
    Write-EnhancedJson "$run/enhanced_cascade.json" $recall
    Invoke-EnhancedPython @('-m','src.v2.enhanced_recommend','--config',"$run/enhanced_cascade.json",'--evaluate-valid','--max-users',"$RuntimeUsers",'--out',"$run/runtime_valid.json")
    Invoke-EnhancedPython @('-m','src.v2.recommend','--config',"$run/recall.json",'--evaluate-valid','--max-users',"$RuntimeUsers",'--compare-without-coarse','--out',"$run/runtime_baseline.json")
    Invoke-EnhancedPython @('-m','src.v2.enhanced_report','--run',$run,'--pool',"$run/candidates",'--out',"$run/summary.json")
    Write-Host "Completed: $run. The default recommender was not changed."
} finally { Pop-Location }
