# The full plan (trunk -> 16K and 4K branches) for every architecture on one Windows GPU, each run under
# the crash supervisor. For a quick end-to-end check use  python scripts/smoke.py  instead.
#   powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1 -Hw rtx3060
param([string]$Hw = "rtx3060",
      [string[]]$Archs = @("dense", "dsa", "kda_full", "kda_dsa", "csa"),
      [string[]]$Stages = @("trunk", "s2_16k", "s2_4k"))
Set-Location (Join-Path $PSScriptRoot "..")
foreach ($arch in $Archs) {
    foreach ($stage in $Stages) {
        $run = "${arch}_${stage}_${Hw}"
        python scripts/supervise.py --run_dir "runs/$run" -- python scripts/train.py --config "configs/$Hw/$stage.yaml" --arch $arch
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "$run ended with code $LASTEXITCODE - see runs/$run/metrics/events.jsonl"
            if ($stage -eq "trunk") { break }
        }
    }
}
$trunk = Get-ChildItem -Directory runs | Where-Object { $_.Name -like "*_trunk_$Hw" } | ForEach-Object { $_.FullName }
$s2 = Get-ChildItem -Directory runs | Where-Object { $_.Name -like "*_s2_*_$Hw" } | ForEach-Object { $_.FullName }
if ($trunk) { python scripts/analyze_runs.py @trunk --out "reports/${Hw}_trunk" }
if ($s2) { python scripts/analyze_runs.py @s2 --out "reports/${Hw}_stage2" }
