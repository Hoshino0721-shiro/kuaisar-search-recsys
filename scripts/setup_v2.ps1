param(
    [string]$BasePython = 'python',
    [switch]$SkipInstall
)
$ErrorActionPreference = 'Stop'
$setupRoot = Split-Path -Parent $PSScriptRoot
$setupPython = Join-Path $setupRoot '.venv/Scripts/python.exe'
$savedTemp = $env:TEMP
$savedTmp = $env:TMP
function Invoke-SetupPython {
    param([string]$Executable, [string[]]$PythonArguments)
    & $Executable @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Environment setup/check failed: $PythonArguments" }
}
Push-Location $setupRoot
try {
    # Keep temporary fixture files within this workspace, including on restricted Windows hosts.
    $setupTemp = Join-Path $setupRoot 'artifacts/environment_tmp'
    New-Item -ItemType Directory -Force -Path $setupTemp | Out-Null
    $env:TEMP = $setupTemp
    $env:TMP = $setupTemp
    if (-not (Test-Path -LiteralPath $setupPython)) {
        Invoke-SetupPython -Executable $BasePython -PythonArguments @('-c', 'import sys; assert sys.version_info >= (3, 10), "Use Python 3.10+; Python 3.12 is recommended"')
        Invoke-SetupPython -Executable $BasePython -PythonArguments @('-m', 'venv', '.venv')
    }
    Invoke-SetupPython -Executable $setupPython -PythonArguments @('-c', 'import sys; assert sys.version_info >= (3, 10), "Existing .venv requires Python 3.10+"; print(sys.version)')
    if (-not $SkipInstall) {
        Invoke-SetupPython -Executable $setupPython -PythonArguments @('-m', 'pip', 'install', '-r', 'requirements-v2-cpu.lock', '--extra-index-url', 'https://download.pytorch.org/whl/cpu')
    }
    Invoke-SetupPython -Executable $setupPython -PythonArguments @('-m', 'pip', 'check')
    Invoke-SetupPython -Executable $setupPython -PythonArguments @('-m', 'unittest', 'discover', '-s', 'tests', '-v')
} finally {
    $env:TEMP = $savedTemp
    $env:TMP = $savedTmp
    Pop-Location
}
