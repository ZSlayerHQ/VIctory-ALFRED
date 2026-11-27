# NVIDIA driver events for one run window, one line each, for run-world.sh to append to its log.
# usage: driver-events.ps1 <start> <end>   (ISO time or Unix seconds, local time)
# Prints nothing and exits 0 when there are no events or the System log cannot be read.
param(
  [Parameter(Mandatory = $true)][string]$Start,
  [Parameter(Mandatory = $true)][string]$End
)

$ErrorActionPreference = 'Stop'

function Convert-ToLocalTime([string]$Value) {
  $epoch = 0L
  if ([long]::TryParse($Value, [ref]$epoch)) {
    return [DateTimeOffset]::FromUnixTimeSeconds($epoch).LocalDateTime
  }
  return [DateTime]::Parse($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

$devicePrefix = [regex]::Escape('\Device\')
$events = @()
try {
  $from = Convert-ToLocalTime $Start
  $to = Convert-ToLocalTime $End
  $events = @(Get-WinEvent -FilterHashtable @{
      LogName = 'System'
      ProviderName = 'nvlddmkm'
      StartTime = $from
      EndTime = $to
    })
} catch {
  exit 0
}

foreach ($event in $events) {
  try {
    $parts = @()
    if ($event.Message) {
      $parts += $event.Message
    } else {
      $xml = [xml]$event.ToXml()
      foreach ($node in $xml.Event.EventData.ChildNodes) {
        if ($node.Name -eq 'Data') { $parts += $node.InnerText }
      }
      # Data[0] is the device path (\Device\Video3); the text is the field that follows it.
      if ($parts.Count -gt 1 -and $parts[0] -match ('^' + $devicePrefix)) {
        $parts = $parts[1..($parts.Count - 1)]
      }
    }
    $text = ($parts -join ' ') -replace '\s+', ' '
    $text = $text -replace ($devicePrefix + '\S+'), ''
    # Trailing runs such as "0 0 0 0" are padding. A register=value field is not a bare
    # number, so the ESR values survive for the summary's distinct value sets.
    $text = $text -replace '(?:[:\s]+(?:0x[0-9a-fA-F]+|\d+))+\s*$', ''
    $text = $text.Trim()
    $time = $event.TimeCreated.ToUniversalTime().ToString('o')
    "driver event: $time id=$($event.Id) $text".TrimEnd()
  } catch {
  }
}
exit 0
