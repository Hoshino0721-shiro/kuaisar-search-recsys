param(
  [string]$Url = 'https://zenodo.org/records/8181109/files/KuaiSAR.zip?download=1',
  [string]$OutFile = 'data/raw/KuaiSAR.zip',
  [int]$ChunkBytes = 8388608,
  [long]$TotalBytes = 488077651,
  [int]$Retries = 5
)

$ErrorActionPreference = 'Stop'
function Assert-Checksum([string]$Path) {
  if ((Get-FileHash -LiteralPath $Path -Algorithm MD5).Hash.ToLowerInvariant() -ne 'daea8cbf605db6bd5841740f0e4a12d9') {
    throw "MD5 mismatch; preserved file for inspection: $Path"
  }
}
if ($ChunkBytes -le 0 -or $Retries -le 0) { throw 'ChunkBytes and Retries must be positive' }
$assembly = [System.Reflection.Assembly]::LoadWithPartialName('System.Net.Http')
if ($null -eq $assembly) { Add-Type -AssemblyName System.Net.Http }
$target = [System.IO.Path]::GetFullPath($OutFile)
$partial = "$target.partial"
$dir = Split-Path -Parent $target
New-Item -ItemType Directory -Path $dir -Force | Out-Null

$existing = 0L
if (Test-Path -LiteralPath $target) {
  $finalLength = (Get-Item -LiteralPath $target).Length
  if ($finalLength -eq $TotalBytes) {
    Assert-Checksum $target
    Write-Host "Already verified: $target ($TotalBytes bytes)"
    exit 0
  }
  throw "Refusing to resume a non-final file at $target ($finalLength bytes). Remove or rename it after inspection."
}
if (Test-Path -LiteralPath $partial) {
  $existing = (Get-Item -LiteralPath $partial).Length
}

$client = New-Object System.Net.Http.HttpClient
$client.DefaultRequestHeaders.UserAgent.ParseAdd('kuaisar-search-recsys/0.1')
$total = $TotalBytes
if ($existing -gt $total) { throw "Existing file is larger than expected: $existing > $total" }
if ($existing -eq $total) {
  Assert-Checksum $partial
  Move-Item -LiteralPath $partial -Destination $target
  Write-Host "Complete: $target"
  exit 0
}

while ($existing -lt $total) {
  $end = [Math]::Min($existing + $ChunkBytes - 1, $total - 1)
  $done = $false
  for ($attempt = 1; $attempt -le $Retries -and -not $done; $attempt++) {
    $request = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Get, $Url)
    $request.Headers.Range = New-Object System.Net.Http.Headers.RangeHeaderValue($existing, $end)
    $response = $null
    try {
      $response = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
      $response.EnsureSuccessStatusCode() | Out-Null
      $range = $response.Content.Headers.ContentRange
      if ([int]$response.StatusCode -ne 206 -or $null -eq $range -or $range.From -ne $existing -or $range.To -ne $end -or $range.Length -ne $total) {
        throw 'Server ignored or returned an inconsistent byte range; nothing appended'
      }
      $bytes = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
      if ($bytes.LongLength -ne ($end - $existing + 1)) { throw 'Incomplete chunk; nothing appended' }
      $file = [System.IO.File]::Open($partial, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
      try {
        $file.Seek($existing, [System.IO.SeekOrigin]::Begin) | Out-Null
        $file.Write($bytes, 0, $bytes.Length)
        $file.Flush()
        $existing += $bytes.Length
      } finally { $file.Dispose() }
      $done = $true
    } catch {
      if ($attempt -eq $Retries) { throw }
      Start-Sleep -Seconds ([Math]::Min(30, 2 * $attempt))
    } finally { if ($null -ne $response) { $response.Dispose() }; $request.Dispose() }
  }
  Write-Host ("Downloaded {0:N0}/{1:N0} bytes ({2:P1})" -f $existing, $total, ($existing / $total))
}

$client.Dispose()
if ((Get-Item -LiteralPath $partial).Length -ne $total) { throw "Downloaded byte count does not match expected total." }
Assert-Checksum $partial
Move-Item -LiteralPath $partial -Destination $target
Write-Host "Complete: $target"
