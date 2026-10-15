$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$fixture = Join-Path $root 'tests/fixtures'
New-Item -ItemType Directory -Force -Path $fixture | Out-Null
@"
user_id,item_id,time,click
u1,i1,2023-05-22T00:00:00Z,1
u1,i2,2023-05-23T00:00:00Z,0
u2,i1,2023-05-24T00:00:00Z,1
u2,i3,2023-05-25T00:00:00Z,0
u3,i4,2023-05-26T00:00:00Z,1
u3,i5,2023-05-27T00:00:00Z,0
u4,i6,2023-05-28T00:00:00Z,1
u4,i7,2023-05-29T00:00:00Z,0
u5,i8,2023-05-30T00:00:00Z,1
u5,i9,2023-05-31T00:00:00Z,0
"@ | Set-Content -Encoding utf8 (Join-Path $fixture 'rec_inter.csv')
Push-Location $root
python -m src.data_pipeline --input-dir tests/fixtures --output-dir $env:TEMP/kuaisar_processed_fixture --summary $env:TEMP/kuaisar_fixture_split_summary.json --train-end 2023-05-30T00:00:00Z --valid-end 2023-05-31T00:00:00Z
if ($LASTEXITCODE -ne 0) { throw 'Pipeline smoke test failed' }
python -c "from src.metrics import auc, ndcg_at_k; assert round(auc([0,1],[.1,.9]), 6)==1.0; assert round(ndcg_at_k(['i1'], ['i1'], 1), 6)==1.0; print('metric smoke test passed')"
if ($LASTEXITCODE -ne 0) { throw 'Metric smoke test failed' }
Pop-Location
