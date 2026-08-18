<#
    start_scheduler.ps1
    -------------------
    Boot wrapper for the nightly scheduler daemon (src/scheduler.py).

    The daemon sleeps until 00:00 and then runs, in order:

      1. src/extract_demand_details.py  - the FULL 36-month pull (the
         self-healing baseline; the dashboard's button does the fast
         incremental one).
      2. src/extract_warehouse_projections.py - the five regional files.
      3. src/extract_key_skus.py - the "Key SKUs" watchlist.
      4. python -m agent.batch - all five models backtested across every view,
         which is what Optimized Projections reads. Runs only if step 1
         succeeded, and also warms the forecast cache for the morning.

    Register it ONCE so it comes back after a reboot (run elevated, and replace
    the path and user to match this host):

      schtasks /create /tn "SmartDemandPlanner Scheduler" ^
        /tr "powershell -NoProfile -ExecutionPolicy Bypass -File \"<repo>\start_scheduler.ps1\"" ^
        /sc onstart /ru <DOMAIN\user> /rp * /rl highest

    Check on it with:  schtasks /query /tn "SmartDemandPlanner Scheduler" /v /fo list
    Stop it with:      schtasks /end   /tn "SmartDemandPlanner Scheduler"

    This is a *supervisor* entry, not the schedule itself: the 00:00 trigger
    lives in scheduler.py, so changing the time is a flag or an env var
    (DEMAND_SCHEDULE_HOUR / DEMAND_SCHEDULE_MINUTE), not a Task Scheduler edit.

    The daemon writes to logs/<yyyy-MM-dd>/logs_scheduler.txt, and each step
    writes to the same per-pull log the dashboard's refresh buttons use. A failed
    step leaves the same .failed marker, so the dashboard's red banner reports a
    bad night without anyone reading a log.

    The interpreter defaults to the repo's Python but can be overridden with the
    DEMAND_PYTHON environment variable (e.g. to point at a venv).
#>

$ErrorActionPreference = 'Stop'

# Anchor to this script's own folder (the repo root), so the task works
# regardless of the working directory Task Scheduler launches it from.
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# Resolve the interpreter for THIS host. The fallback used to be one developer's
# hardcoded per-user path, which simply does not exist on the dev server that
# also runs this repo off the same network share — the task would have died at
# launch there with nothing but a Task Scheduler exit code. Prefer the explicit
# override, then that known path if it happens to be present, then whatever
# python is on PATH; fail loudly rather than run nothing.
$Python = $env:DEMAND_PYTHON
if ([string]::IsNullOrWhiteSpace($Python) -or -not (Test-Path $Python)) {
    $Candidates = @(
        'C:\Users\engo\AppData\Local\Python\pythoncore-3.14-64\python.exe',
        'C:\Python313\python.exe'
    )
    $Python = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $Python) {
        $OnPath = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($OnPath) { $Python = $OnPath.Source }
    }
}
if (-not $Python -or -not (Test-Path $Python)) {
    throw ("No Python interpreter found on $env:COMPUTERNAME. Set the " +
           "DEMAND_PYTHON environment variable to the interpreter that has " +
           "this project's requirements installed.")
}

# Unbuffered, UTF-8: the daemon runs for months with its stdout redirected, so
# block buffering would hold the startup banner back indefinitely, and the
# locale code page would mangle the em-dashes in its log lines.
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

$Scheduler = Join-Path $Root 'src\scheduler.py'

# Exec in place: Task Scheduler supervises the python process directly, so
# "Last Run Result" reflects the daemon and stopping the task stops it.
& $Python $Scheduler @args
exit $LASTEXITCODE
