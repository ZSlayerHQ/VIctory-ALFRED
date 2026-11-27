# The diagnostic tool pack

**Established tools are the judges; this project's own diagnostics are glue.** A number that decides anything is
confirmed against one of the tools below, and a speed or picture claim in a pull request names the tool that measured
it. Reason: these tools have years of use behind them, and in this project's record three of its own measuring tools
misreported within a day of being written.

Everything here is for Windows 10 or 11 on x64 with a Direct3D 12 card. `node harness/tools-check.mjs` prints one
line per tool (found or missing, its version, where it looked) and exits 0 when the required pair, PresentMon and
ffmpeg, is ready for the harness.

## Before you install

- An agent runs the steps marked **agent**. A step marked **human** needs administrator rights: Windows raises a
  prompt only a person can answer, so ask your human to run it.
- Check a downloaded program or installer before running it. `Status` should be `Valid` and `Signer` the vendor:
  `Get-AuthenticodeSignature .\file.exe | Select-Object Status, @{n='Signer'; e={$_.SignerCertificate.Subject}}`.
  winget checks each download against the hash its package entry lists; for a build without a signature, compare the
  vendor's published checksum with `Get-FileHash`.
- Captures, traces and frames from the title stay on your machine: never attach them to an issue or commit them
  ([AGENTS.md](../AGENTS.md), "Hard rules"; [the asset policy](../victory/docs/asset-policy.md)). Post the numbers.

## PresentMon 2.6.0, console application: frame time and pacing

- Install (**agent**): `winget install --id Intel.PresentMon.Console -e`, or `PresentMon-2.6.0-x64.exe` from
  [the release page](https://github.com/GameTechDev/PresentMon/releases/tag/v2.6.0). Then set `PRESENTMON_EXE` in
  `harness/local.env` to that file (`(Get-Command presentmon).Source` prints winget's); the harness reads only that.
- Rights (**human**): capturing needs administrator rights or membership of Windows' Performance Log Users group
  ([PresentMon's README](https://github.com/GameTechDev/PresentMon/blob/main/README.md)).
- Check: the executable is signed by Intel Corporation; tools-check shows PresentMon found.
- The harness reads `MsBetweenPresents` and `MsGPUBusy` (or `FrameTime` and `GPUBusy`, the newer names);
  `MsCPUBusy` is in the same file for a processor-side claim.

## Windows Performance Toolkit (`wpr`, `xperf`, WPA): where a thread's time goes, lock waits, system calls

- `wpr` is part of Windows 10 and later. `xperf` and the analyzer (WPA) come with the Windows ADK.
- Install (**human**): the installer from [the ADK page](https://learn.microsoft.com/windows-hardware/get-started/adk-install),
  run as administrator, with Windows Performance Toolkit selected. winget's `Microsoft.WindowsADK` installs the
  newest ADK, which that page lists for one Windows release only; pick the version the page gives for yours. WPA
  alone is also on the Microsoft Store as Windows Performance Analyzer (Preview).
- Rights (**human**): recording with `wpr` runs from an administrator prompt.
- Keep traces out of the repository and out of issues: a system-wide trace holds every process on the machine.
- Check: tools-check shows the toolkit found, with `xperf.exe` beside the analyzer. Used here: WPA 11.7.

## Tracy 0.14.1: a live timeline of threads, locks and GPU zones

- The runtime's Tracy integration is planned, not merged: there is nothing to connect the viewer to yet.
- Install (**agent**): `windows-0.14.1.zip` from [the release page](https://github.com/wolfpld/tracy/releases/tag/v0.14.1);
  unzip it and put its folder on `PATH`. The viewer is `tracy-profiler.exe`.
- Use the viewer of the same version as the client: the network protocol can change between releases, and a
  mismatched viewer cannot connect ([the manual](https://github.com/wolfpld/tracy/blob/v0.14.1/manual/tracy.md)).
- Build a client without Tracy's crash handler and without code transfer: in C++, `TRACY_NO_CRASH_HANDLER` and
  `TRACY_NO_CODE_TRANSFER`; with the Rust `tracy-client` crate, `default-features = false`, then list the features
  you want, leaving out `crash-handler` and `code-transfer` (both are on by default). Reason: code transfer copies
  the running program's machine code into the trace, and here that program is the title.

## RenderDoc 1.47: one frame, every draw, texture and state

- Install (**agent**): the portable `RenderDoc_1.47_64.zip` from [the builds page](https://renderdoc.org/builds), unzipped;
  or (**human**) `winget install --id BaldurKarlsson.RenderDoc -e`, a machine-wide installer.
- Set `RENDERDOC_DIR` in `harness/local.env` to the folder that holds `renderdoccmd.exe`. The harness captures with
  `--renderdoc --capture-after-ms <ms>`.
- Check: tools-check shows RenderDoc found. Used here: 1.47.

## PIX on Windows: Direct3D 12 timing captures, passes and shaders

- Install (**human**): `winget install --id Microsoft.PIX -e`, a machine-wide installer; or
  [the download page](https://devblogs.microsoft.com/pix/download/). `WinPix.exe` and `pixtool.exe` land in
  `%ProgramFiles%\Microsoft PIX\<version>\`.
- Check: tools-check shows PIX found. Used here: 2611.19.

## NVIDIA Nsight Graphics, GPU Trace: GPU time per pass and per shader

- NVIDIA cards only (GPU Trace profiles Turing and later).
- Install (**human**): the installer from [the download page](https://developer.nvidia.com/nsight-graphics); there is
  no winget package. Whether the download needs a developer-program sign-in: check the vendor's page.
- The harness traces with `--gpu-trace --trace-after-ms <ms>`; it finds `ngfx.exe` through `VICTORY_NGFX`, else the
  newest `Nsight Graphics <version>` folder under `C:\Program Files\NVIDIA Corporation`.
- Check: tools-check shows Nsight Graphics found. Used here: 2026.4.0.

## ffmpeg: picture checks (`blackframe`, `freezedetect`, `ssim`, `psnr`)

- Install (**agent**): `winget install --id Gyan.FFmpeg -e`, which puts `ffmpeg` and `ffprobe` on `PATH`; or a
  portable build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), checked against its `.sha256` file, with its
  `bin` folder on `PATH`. `harness/picture-check.mjs` calls both by name.
- Check: `ffmpeg -hide_banner -filters` lists the four filters; tools-check shows ffmpeg found. Used here: 8.1.
