param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$repoRootNormalized = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\')
$repoRootRegex = [Regex]::Escape($repoRootNormalized)

$allProcesses = Get-CimInstance Win32_Process

$auraTestProcesses = $allProcesses | Where-Object {
    $_.ExecutablePath -match "^${repoRootRegex}\\target\\debug\\deps\\aura-.*\.exe$"
}

$cargoTestProcesses = $allProcesses | Where-Object {
    $_.Name -eq "cargo.exe" -and
    $_.CommandLine -like "* test *" -and
    $_.ProcessId -in $auraTestProcesses.ParentProcessId
}

$testProcesses = @($auraTestProcesses) + @($cargoTestProcesses)

if (-not $testProcesses) {
    Write-Output "No matching cargo/aura test processes found under $repoRootNormalized"
    exit 0
}

$ids = $testProcesses.ProcessId | Sort-Object -Unique
Write-Output ("Stopping test-lock processes: " + ($ids -join ", "))

foreach ($id in $ids) {
    try {
        Stop-Process -Id $id -Force -ErrorAction Stop
    } catch {
        Write-Warning ("Failed to stop process {0}: {1}" -f $id, $_.Exception.Message)
    }
}

Start-Sleep -Milliseconds 300

$remaining = Get-Process -Id $ids -ErrorAction SilentlyContinue
if ($remaining) {
    $remainingIds = $remaining.Id | Sort-Object -Unique
    Write-Warning ("Processes still running: " + ($remainingIds -join ", "))
    exit 1
}

Write-Output "Windows test-lock cleanup completed."
