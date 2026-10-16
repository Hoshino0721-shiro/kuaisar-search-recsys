param(
  [string]$Python = '.venv/Scripts/python.exe',
  [int]$MaxUsers = 256,
  [string]$RuntimeDir = 'artifacts/v2_runtime',
  [switch]$SkipDownload
)
$ErrorActionPreference = 'Stop'
$v2Root = Split-Path -Parent $PSScriptRoot
function Invoke-V2Python([string[]]$Arguments) {
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) { throw "V2 stage failed: $Arguments" }
}
Push-Location $v2Root
try {
  if (-not (Test-Path -LiteralPath 'data/raw/KuaiSAR.zip')) {
    if ($SkipDownload) { throw 'Official ZIP is missing' }
    & "$PSScriptRoot/download_kuaisar.ps1"
    if (-not (Test-Path -LiteralPath 'data/raw/KuaiSAR.zip')) { throw 'Download did not produce the official ZIP' }
  }
  Invoke-V2Python -Arguments @('-m','unittest','discover','-s','tests','-v')
  if (-not (Test-Path -LiteralPath 'data/processed/kuaisar_small_v1/manifest.json')) {
    Invoke-V2Python -Arguments @('-m','src.v2.prepare_data')
  }
  if (-not (Test-Path -LiteralPath 'data/processed/kuaisar_small_v1/ctr_features_v1/ready.json')) {
    Invoke-V2Python -Arguments @('-m','src.ctr_features')
  }
  $v2Coarse = "$RuntimeDir/coarse_lr"
  $v2Fine = "$RuntimeDir/fine_dcn"
  if (-not (Test-Path -LiteralPath "$v2Coarse/model.pt")) {
    Invoke-V2Python -Arguments @('-m','src.v2.train_rankers','--kind','coarse_lr','--out',$v2Coarse)
  }
  if (-not (Test-Path -LiteralPath "$v2Fine/model.pt")) {
    Invoke-V2Python -Arguments @('-m','src.v2.train_rankers','--kind','fine_dcn','--out',$v2Fine)
  }
  $v2Config = Get-Content -LiteralPath 'configs/cascade_v2.json' -Raw | ConvertFrom-Json
  $v2Config.coarse_model = "$v2Coarse/model.pt"
  $v2Config.fine_model = "$v2Fine/model.pt"
  $v2ConfigPath = "$RuntimeDir/cascade.json"
  [System.IO.File]::WriteAllText((Join-Path (Get-Location).Path $v2ConfigPath), ($v2Config | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
  $v2Evaluation = "$RuntimeDir/evaluation_${MaxUsers}.json"
  if (-not (Test-Path -LiteralPath $v2Evaluation)) {
    Invoke-V2Python -Arguments @('-m','src.v2.recommend','--config',$v2ConfigPath,'--evaluate-valid','--max-users',"$MaxUsers",'--compare-without-coarse','--out',$v2Evaluation)
  }
} finally { Pop-Location }
