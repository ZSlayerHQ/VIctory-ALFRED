# Waits up to 120 s for this run's game window and prints "<pid> <main window handle>" once, or nothing.
# The selection is run-process.ps1 -Strict's: only a victory.exe created after this run began, with " run " and the
# game's path on its command line, never a fallback. Queries only: it starts, stops and changes nothing (G-016).
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
  $pids = & "$PSScriptRoot/run-process.ps1" -Strict -Pids
  foreach ($id in @($pids)) {
    if (-not $id) { continue }
    $p = Get-Process -Id ([int]$id) -ErrorAction SilentlyContinue
    if ($p -and $p.MainWindowHandle -ne [IntPtr]::Zero) {
      "{0} {1}" -f $p.Id, $p.MainWindowHandle.ToInt64()
      exit 0
    }
  }
  Start-Sleep -Milliseconds 1000
}
exit 0
