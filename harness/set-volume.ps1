# Sets this run's game to a quiet volume in Windows' own mixer and says what it did (one line per change).
# Without it the title plays at full volume.
# Only the game's own audio sessions are touched: by process id, never by name, on every active playback device.
# It keeps looking until the game exits, because the title opens its audio some time after its window, and sets
# each session once: a level changed by hand in the mixer afterwards stays as the person set it.
# It acts only while the process it picked still holds that id: the process's start time is read when it is
# picked and checked before every pass. A game that ends leaves its id free, and Windows gives ids out again, so
# without the check another program's sound (OBS, say) could be changed.
# -ProcessId names the process; without it the process is this run's game, by run-process.ps1's strict rule
# (created after this run began; no fallback, G-016). -List prints the sessions and changes nothing.
param([int]$Percent = 4, [int]$ProcessId = 0, [switch]$List, [switch]$Once)

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace RunVolume {
  [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
  class MMDeviceEnumerator { }

  [Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IMMDeviceEnumerator {
    [PreserveSig] int EnumAudioEndpoints(int dataFlow, int stateMask, out IMMDeviceCollection devices);
  }

  [Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IMMDeviceCollection {
    [PreserveSig] int GetCount(out int count);
    [PreserveSig] int Item(int index, out IMMDevice device);
  }

  [Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IMMDevice {
    [PreserveSig] int Activate(ref Guid iid, int context, IntPtr activation,
      [MarshalAs(UnmanagedType.IUnknown)] out object instance);
  }

  // IAudioSessionManager's two methods come first in the table; they are not called here.
  [Guid("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IAudioSessionManager2 {
    [PreserveSig] int GetAudioSessionControl(IntPtr a, int b, out IntPtr c);
    [PreserveSig] int GetSimpleAudioVolume(IntPtr a, int b, out IntPtr c);
    [PreserveSig] int GetSessionEnumerator(out IAudioSessionEnumerator sessions);
  }

  [Guid("E2F5BB11-0570-40CA-ACDD-3AA01277DEE8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IAudioSessionEnumerator {
    [PreserveSig] int GetCount(out int count);
    [PreserveSig] int GetSession(int index, out IAudioSessionControl2 session);
  }

  // IAudioSessionControl's nine methods, then two of IAudioSessionControl2's, come before GetProcessId.
  [Guid("BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IAudioSessionControl2 {
    [PreserveSig] int GetState(out int state);
    [PreserveSig] int GetDisplayName(out IntPtr name);
    [PreserveSig] int SetDisplayName(IntPtr name, IntPtr context);
    [PreserveSig] int GetIconPath(out IntPtr path);
    [PreserveSig] int SetIconPath(IntPtr path, IntPtr context);
    [PreserveSig] int GetGroupingParam(out Guid group);
    [PreserveSig] int SetGroupingParam(IntPtr group, IntPtr context);
    [PreserveSig] int RegisterAudioSessionNotification(IntPtr client);
    [PreserveSig] int UnregisterAudioSessionNotification(IntPtr client);
    [PreserveSig] int GetSessionIdentifier(out IntPtr id);
    [PreserveSig] int GetSessionInstanceIdentifier(out IntPtr id);
    [PreserveSig] int GetProcessId(out uint pid);
  }

  [Guid("87CE5498-68D6-44E5-9215-6DA47EF883D8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface ISimpleAudioVolume {
    [PreserveSig] int SetMasterVolume(float level, ref Guid context);
    [PreserveSig] int GetMasterVolume(out float level);
  }

  public static class Mixer {
    // Sessions already set once: a level a person changes afterwards in the mixer is left alone.
    static readonly HashSet<string> Done = new HashSet<string>();

    // One line per session of `pid` on any active playback device. When `change` is set, a session not seen
    // before whose level differs from `level` is set, once; the line says the level before and after.
    public static List<string> Sessions(uint pid, float level, bool change) {
      var lines = new List<string>();
      var enumerator = (IMMDeviceEnumerator)(new MMDeviceEnumerator());
      IMMDeviceCollection devices;
      if (enumerator.EnumAudioEndpoints(0, 1, out devices) != 0) return lines;
      int deviceCount;
      devices.GetCount(out deviceCount);
      Guid managerId = typeof(IAudioSessionManager2).GUID;
      for (int d = 0; d < deviceCount; d++) {
        IMMDevice device;
        if (devices.Item(d, out device) != 0) continue;
        object instance;
        if (device.Activate(ref managerId, 23, IntPtr.Zero, out instance) != 0) continue;
        IAudioSessionEnumerator sessions;
        if (((IAudioSessionManager2)instance).GetSessionEnumerator(out sessions) != 0) continue;
        int count;
        sessions.GetCount(out count);
        for (int i = 0; i < count; i++) {
          IAudioSessionControl2 session;
          if (sessions.GetSession(i, out session) != 0) continue;
          uint owner;
          if (session.GetProcessId(out owner) != 0 || owner != pid) continue;
          var volume = session as ISimpleAudioVolume;
          if (volume == null) continue;
          string key = d + ":" + i;
          IntPtr id;
          if (session.GetSessionInstanceIdentifier(out id) == 0 && id != IntPtr.Zero) {
            key = Marshal.PtrToStringUni(id);
            Marshal.FreeCoTaskMem(id);
          }
          if (change && !Done.Add(key)) continue;
          float before;
          volume.GetMasterVolume(out before);
          bool differs = Math.Abs(before - level) > 0.005f;
          if (change && differs) {
            Guid none = Guid.Empty;
            volume.SetMasterVolume(level, ref none);
            lines.Add(String.Format("device {0}: {1:0}% -> {2:0}%", d, before * 100, level * 100));
          } else if (!change) {
            lines.Add(String.Format("device {0}: {1:0}%", d, before * 100));
          }
        }
      }
      return lines;
    }
  }
}
'@

if ($Percent -lt 0 -or $Percent -gt 100) { "volume: $Percent is not a percentage; nothing changed"; exit 1 }

$deadline = (Get-Date).AddSeconds(120)
while ($ProcessId -eq 0 -and (Get-Date) -lt $deadline) {
  $picked = & {
    . "$PSScriptRoot/run-process.ps1"
    @((Get-RunVictorySelection -Strict).Process | ForEach-Object { $_.ProcessId })
  }
  if ($picked.Count -gt 0) { $ProcessId = [int]$picked[0] } else { Start-Sleep -Seconds 2 }
}
if ($ProcessId -eq 0) { "volume: this run's game was not found; nothing changed"; exit 0 }
$picked = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
if (-not $picked) { "volume: pid $ProcessId is gone; nothing changed"; exit 0 }
$pickedStart = $picked.StartTime
# Whether the process holding the id is still the one picked (same start time).
function Test-SameProcess {
  $now = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
  return [bool]($now -and $now.StartTime -eq $pickedStart)
}

$level = [single]($Percent / 100.0)
if ($List) {
  $lines = [RunVolume.Mixer]::Sessions([uint32]$ProcessId, $level, $false)
  if ($lines.Count -eq 0) { "volume: pid $ProcessId has no audio session yet" }
  else { $lines | ForEach-Object { "volume: pid $ProcessId, $_" } }
  exit 0
}

$said = $false
do {
  if (-not (Test-SameProcess)) { break }
  foreach ($line in [RunVolume.Mixer]::Sessions([uint32]$ProcessId, $level, $true)) {
    "volume: pid $ProcessId, $line"
    $said = $true
  }
  if ($Once) { break }
  Start-Sleep -Seconds 3
} while (Test-SameProcess)
if (-not $said) { "volume: pid $ProcessId opened no audio session; nothing changed" }
