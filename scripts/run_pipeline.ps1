param([switch]$VerifyOnly)
$ErrorActionPreference = 'Stop'
$env:LOKY_MAX_CPU_COUNT = '8'
$projectRoot = Split-Path -Parent $PSScriptRoot
function Invoke-ProjectPython([string[]]$Arguments) {
  & python @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Python stage failed: $Arguments" }
}
function Invoke-Stage([string]$ResultPath, [string[]]$Arguments) {
  if (Test-Path -LiteralPath $ResultPath) { Write-Host "Using existing result: $ResultPath" }
  else { Invoke-ProjectPython $Arguments }
}
Push-Location $projectRoot
try {
  Invoke-ProjectPython @('-m','unittest','discover','-s','tests','-v')
  if (-not $VerifyOnly) {
    Invoke-Stage 'data/processed/kuaisar_small_v1/manifest.json' @('-m','src.prepare_data')
    Invoke-Stage 'experiments/r0_p0_valid/metrics.json' @('-m','src.baseline')
    Invoke-Stage 'experiments/r1_itemcf_valid/metrics.json' @('-m','src.itemcf')
    Invoke-Stage 'experiments/r1_itemcf_full_valid/metrics.json' @('-m','src.itemcf','--full-cooccurrence')
    Invoke-Stage 'experiments/r2_two_tower_valid/metrics.json' @('-m','src.two_tower')
    Invoke-Stage 'experiments/r2_history_attributes_valid/metrics.json' @('-m','src.history_tower')
    Invoke-Stage 'experiments/r2_history_only_valid/metrics.json' @('-m','src.history_tower','--no-attributes','--run','r2_history_only_valid')
    Invoke-Stage 'experiments/r2_history_attributes_valid/audit.json' @('-m','src.audit_two_tower','--run','r2_history_attributes_valid')
    Invoke-Stage 'experiments/r2_history_only_valid/audit.json' @('-m','src.audit_two_tower','--run','r2_history_only_valid')
    Invoke-Stage 'data/processed/kuaisar_small_v1/ctr_features_v1/ready.json' @('-m','src.ctr_features')
    Invoke-Stage 'experiments/p_rank_valid/metrics.json' @('-m','src.ranking')
    Invoke-Stage 'experiments/pipeline_valid/metrics.json' @('-m','src.retrieval_pipeline')
    Invoke-Stage 'experiments/search_valid/metrics.json' @('-m','src.search_ranking')
    Invoke-Stage 'experiments/ann_valid/metrics.json' @('-m','src.benchmark_ann')
    Invoke-Stage 'experiments/final_test/metrics.json' @('-m','src.final_evaluate')
  }
  Invoke-ProjectPython @('-m','src.verify_release')
  Invoke-ProjectPython @('-m','src.recommend','--user-id','20235','--top-k','10')
} finally { Pop-Location }
