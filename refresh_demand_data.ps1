<#
    refresh_demand_data.ps1
    -----------------------
    Wrapper for the nightly scheduled pull of BOTH dashboard data sets from the
    SQL data warehouse:

      1. src/extract_demand_details.py (the ~10-minute batch) writes
         all_demand_projections_<date>.xlsx atomically into the folder the
         dashboard discovers snapshots in (raw_inputs/demand_projections).
      2. src/extract_warehouse_projections.py (a ~2-minute batch) writes the
         five regional AU/CA/EU/JP/US_warehouse_projections_<date>.xlsx files
         (raw_inputs/warehouse_projections) that drive the missing-projections
         table.

    The dashboard then serves the pre-computed files instantly — nobody waits.

    The demand run deliberately does the FULL 36-month pull (no --incremental
    flag): it is the self-healing baseline that picks up restated actuals, item
    renames and customer remaps. The dashboard's refresh buttons run the fast
    incremental demand pull / the same warehouse pull on demand.

    The warehouse pull runs even if the demand pull failed (they are
    independent data sets), and the script exits with the worst of the two exit
    codes so Task Scheduler flags a failure in either (Last Run Result).

    Register it with Windows Task Scheduler (see the schtasks command in the
    project notes). It logs each run, with timestamps, to
    logs/<yyyy-MM-dd>/logs_refresh.txt next to this script.

    The interpreter defaults to the repo's Python but can be overridden with the
    DEMAND_PYTHON environment variable (e.g. to point at a venv).
#>

$ErrorActionPreference = 'Stop'

# Anchor everything to this script's own folder (the repo root), so the task
# works regardless of the working directory Task Scheduler launches it from.
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = $env:DEMAND_PYTHON
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = 'C:\Users\engo\AppData\Local\Python\pythoncore-3.14-64\python.exe'
}

$DemandScript    = Join-Path $Root 'src\extract_demand_details.py'
$WarehouseScript = Join-Path $Root 'src\extract_warehouse_projections.py'

# Logs are organized by day: logs/<yyyy-MM-dd>/logs_refresh.txt
$Today  = Get-Date -Format 'yyyy-MM-dd'
$LogDir = Join-Path (Join-Path $Root 'logs') $Today
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log    = Join-Path $LogDir 'logs_refresh.txt'
$Stamp  = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

Add-Content -Path $Log -Value "`n===== scheduled DW refresh started $Stamp ====="

# Run the demand pull, tee stdout+stderr into the log. The extract writes the
# workbook atomically, so the dashboard never reads a half-written file
# mid-refresh.
& $Python $DemandScript *>&1 | Tee-Object -FilePath $Log -Append
$demandCode = $LASTEXITCODE

$Stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -Path $Log -Value "===== scheduled DW refresh finished $Stamp (exit $demandCode) ====="

# Warehouse projections: an independent data set, so it runs regardless of the
# demand pull's outcome. Each of its five regional files is written atomically.
$Stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -Path $Log -Value "`n===== scheduled warehouse refresh started $Stamp ====="

& $Python $WarehouseScript *>&1 | Tee-Object -FilePath $Log -Append
$warehouseCode = $LASTEXITCODE

$Stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -Path $Log -Value "===== scheduled warehouse refresh finished $Stamp (exit $warehouseCode) ====="

# Precompute every view's agent summary in parallel, off the request path, so
# the dashboard serves agent results (best model + backtest + narrative)
# instantly instead of running the slow multi-model backtest live. Reads the
# fresh snapshot the demand pull just wrote (fast, via its Parquet sidecar).
# Run as a module from src/ so `agent.batch` resolves; only if the demand pull
# succeeded (a failed pull leaves no fresh data worth summarizing).
$agentCode = 0
if ($demandCode -eq 0) {
    $Stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $Log -Value "`n===== scheduled agent precompute started $Stamp ====="
    Push-Location (Join-Path $Root 'src')
    try {
        & $Python -m agent.batch *>&1 | Tee-Object -FilePath $Log -Append
        $agentCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    $Stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $Log -Value "===== scheduled agent precompute finished $Stamp (exit $agentCode) ====="
}

# Worst exit code wins, so Task Scheduler flags a failure in any pull/step.
exit ([Math]::Max([Math]::Max([int]$demandCode, [int]$warehouseCode), [int]$agentCode))
