# Bring the Gazebo window (WSLg) onto the main screen. WSLg sometimes leaves it minimized
# off-screen, and clicking its taskbar button does not restore it.
#   powershell -ExecutionPolicy Bypass -File limo\show_gazebo.ps1
param([int]$X = 200, [int]$Y = 100, [int]$Width = 1600, [int]$Height = 1000)

Add-Type @"
using System; using System.Text; using System.Runtime.InteropServices;
public class GzWin {
  public delegate bool EnumProc(IntPtr h, IntPtr p);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr p);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int hh, bool repaint);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
}
"@
$script:found = [IntPtr]::Zero
[GzWin]::EnumWindows({ param($h, $p)
    $sb = New-Object System.Text.StringBuilder 256
    [void][GzWin]::GetWindowText($h, $sb, 256)
    if ($sb.ToString() -match 'Gazebo Sim') { $script:found = $h }
    $true }, [IntPtr]::Zero) | Out-Null
if ($script:found -eq [IntPtr]::Zero) { Write-Output "No Gazebo window: start the sim with gui:=true"; exit 1 }
$title = New-Object System.Text.StringBuilder 256
[void][GzWin]::GetWindowText($script:found, $title, 256)
if ($title.ToString() -match 'COPY MODE') {
    # WSL 2.7.x bug (microsoft/wslg#1456): /mnt/shared_memory is missing and WSLg windows
    # stay invisible. Restarting the WSLg compositor fixes it until the next WSL restart;
    # it closes all Linux windows, so restart the sim with gui:=true afterwards.
    Write-Output "WSLg is in COPY MODE (windows invisible). Fix, then restart the sim:"
    Write-Output "  wsl -d Ubuntu-24.04-ROS2 --system -- sh -c 'kill `$(pgrep -x weston)'"
    exit 2
}
[void][GzWin]::ShowWindow($script:found, 9)   # SW_RESTORE
Start-Sleep -Milliseconds 300
[void][GzWin]::MoveWindow($script:found, $X, $Y, $Width, $Height, $true)
[void][GzWin]::SetForegroundWindow($script:found)
Write-Output "Gazebo window restored at ($X, $Y), ${Width}x${Height}"
