<#
    refresh_demand_data.ps1
    -----------------------
    Wrapper for the nightly scheduled pull of the demand snapshot from the SQL
    data warehouse. Runs extract_demand_details.py (the ~10-minute batch), which
    writes all_demand_projections_<date>.xlsx atomically into the folder the
    dashboard discovers snapshots in (raw_inputs/demand_projections). The
    dashboard then serves that pre-computed file instantly — nobody waits 10 min.

    Register it with Windows Task Scheduler (see the schtasks command in the
    project notes). It logs each run, with timestamps, to logs_refresh.txt next
    to this script, and exits with the extract's own exit code so Task Scheduler
    records success/failure (Last Run Result).

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

$Script = Join-Path $Root 'extract_demand_details.py'
$Log    = Join-Path $Root 'logs_refresh.txt'
$Stamp  = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

Add-Content -Path $Log -Value "`n===== scheduled DW refresh started $Stamp ====="

# Run the pull, tee stdout+stderr into the log. The extract writes the workbook
# atomically, so the dashboard never reads a half-written file mid-refresh.
& $Python $Script *>&1 | Tee-Object -FilePath $Log -Append
$code = $LASTEXITCODE

$Stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -Path $Log -Value "===== scheduled DW refresh finished $Stamp (exit $code) ====="

exit $code
