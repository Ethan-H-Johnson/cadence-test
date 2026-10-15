# Cadence on the AgileX LIMO

One continuing Cadence brain drives the LIMO Pro rover. Development happens in simulation
that is set up to match the rover; the robot comes later.

## The rover, and how the sim matches it

| | Rover (checked 2026-10-15) | Gazebo sim (`ros2/`) |
| --- | --- | --- |
| Computer | Jetson Orin Nano Super, Ubuntu 24.04 | WSL Ubuntu 24.04 (`Ubuntu-24.04-ROS2`) |
| ROS | ROS 2 Jazzy, domain 26, discovery local | ROS 2 Jazzy, domain 26, discovery local |
| Chassis driver | AgileX `limo_base`, `limo_ros2` @ `dcc5a86`, telemetry-only | Gazebo DiffDrive with the driver's kinematics |
| `/odom` | 50 Hz, `odom` -> `base_link` | 50 Hz, `odom` -> `base_link` |
| `/imu` | with orientation | 100 Hz, with orientation |
| `/cmd_vel` | driver subscribes; **writes disabled** | drives the robot |
| Lidar | CP2102 on `/dev/ttyUSB0`; LIMO Pro ships the EAI T-mini Pro; **not configured** | T-mini Pro-like: 10 Hz, 0.9 deg spacing, 0.02-12 m, front 240 deg |
| Motion mode | **unknown** (`/limo_status` `motion_mode`) | four-wheel differential |
| Brain step | **13.2 ms** median (measured, `rover.json`) | emulated with `--profile orin-nano-super` |
| Body | AgileX geometry | AgileX geometry and meshes; the body is drawn as its collision box |

Still to measure on the rover, and how the sim may need to change:

- **Motion mode.** In Ackermann mode the rover cannot spin in place. The action set and the
  sim would then need car-like steering (the driver uses a 28 degree inner angle).
- **Lidar.** Which model it is, its rate, and whether the rover's shell blocks part of the
  scan. AgileX's body mesh wraps the lidar, and in Gazebo it blocked the scan at about
  +-35..50 degrees. The sim therefore draws the body as a box (`body_mesh:=true` brings the
  mesh back).
- **Real dynamics.** Acceleration, top speed and turning on your floor. The sim robot
  follows commands almost exactly; real skid-steer turning will slip.

`rover_check.sh` gathers the first two (read-only) once the lidar driver is running.

```
 Cadence controller (Python 3.11+)  --TCP JSON lines-->  limo_cadence_bridge (ROS 2 node)
   limo_brain.run                                          /scan /odom /imu -> read
   resource budget, toy sim                                cmd -> /cmd_vel (20 Hz, 0.5 s watchdog)
```

The same bridge node runs in the sim and on the rover. The controller can run on Windows
(the sim), on the rover itself (Python 3.12 and NumPy are already there), or on a PC over
the network.

## Layout

| Path | What |
| --- | --- |
| `limo_brain/robot.py` | Observation (8 lidar sectors + speed + turn rate), 5 drive actions, reward, falls, weak-wheel disturbance, a hand-written `reflex` reference controller |
| `limo_brain/budget.py` | Resource profiles: `orin-nano-super` (measured), `jetson-nano`, `pi4`, `pi5` (estimates) |
| `limo_brain/envs.py` | `ToySim` (built-in 2D LIMO, no install) and `Bridge` (Gazebo or rover, paced by the bridge's clock) |
| `limo_brain/run.py` | The controller loop and CLI |
| `limo_brain/bench.py` | Times one brain step; run it on the rover to calibrate |
| `ros2/src/limo_sim` | Gazebo Harmonic LIMO Pro (`urdf/limo_pro.urdf.xacro`), `worlds/arena.sdf`, `launch/sim.launch.py` |
| `ros2/src/limo_cadence_bridge` | ROS 2 bridge node; `launch/robot.launch.py` for the rover |
| `ros2/src/agilex` | AgileX `limo_description` at the rover's commit (BSD-3), fetched by `fetch_limo_ros2.sh`, not in git |
| `setup_wsl_jazzy.sh`, `run_sim_jazzy.sh`, `stop_sim_jazzy.sh` | Install, start and stop the ROS 2 sim |
| `rover_check.sh` | Read-only rover survey and bench |
| `ros/`, `*_wsl*.sh` (noetic) | Legacy ROS1 sim for the original Jetson Nano LIMO (below) |

## Quick start: built-in toy sim (no ROS)

From `limo/`:

```sh
py -m limo_brain.run --env toy --ticks 8000                         # Cadence, this PC's speed
py -m limo_brain.run --env toy --profile orin-nano-super --calibration rover.json   # as the rover
py -m limo_brain.run --env toy --policy reflex                      # reference controllers
py -m limo_brain.run --env toy --weak-left 0.5 --weak-at 4000 --save brain.npz
py -m pytest tests
```

Each run prints a row per `--window` ticks (mean reward, bump share, aroused share, action
counts, overruns) and writes `runs/run-*.json`.

## Resource budget (`--profile`)

| Flag | Effect |
| --- | --- |
| `--profile orin-nano-super` | 6 cores, 3000 MB allowance, slowdown 6.1x (measured against an idle i9-9900K) |
| `--calibration rover.json` | Use the rover's bench directly (preferred; robust to a busy host) |
| `--profile jetson-nano` / `pi4` / `pi5` | Estimates: 10x / 7x / 3x |
| `--slowdown X` | Set the factor directly |
| `--cores N` | CPU affinity and math-library threads |
| `--memory-mb M`, `--hard-memory` | Memory allowance; reported by default, a hard cap on Linux |

Each brain step is timed on the host and multiplied by the slowdown. The previous command
keeps driving the robot for that stretched time, as it would while the rover's computer
thinks, and then the new command is applied. A step longer than the control period (`--hz`,
default 10, the lidar's rate) delays the next tick and counts as an overrun. At startup the
host runs the same bench as `rover.json`, so a busy or faster PC still emulates the rover:
with Gazebo loading this PC, the slowdown came out at 3.25x and the emulated step at 12.4 ms,
against 13.2 ms measured on the rover.

The toy sim is lockstep. The Gazebo/rover client paces itself by the bridge's clock (sim
time in Gazebo), so the emulated latency is exact even when the sim runs slower than real
time. The slowdown models CPU time only, not thermal throttling or the ROS drivers sharing
the rover's cores.

To recalibrate, copy the code to the rover and run the bench there:

```powershell
ssh agilex@192.168.55.1 "mkdir -p ~/cadence_rover/src ~/cadence_rover/limo"
scp -r ..\src\cadence agilex@192.168.55.1:~/cadence_rover/src/
scp -r limo_brain rover_check.sh agilex@192.168.55.1:~/cadence_rover/limo/
ssh agilex@192.168.55.1 "cd ~/cadence_rover/limo && python3 -m limo_brain.bench --save ~/cadence_rover/rover.json"
scp agilex@192.168.55.1:~/cadence_rover/rover.json rover.json
```

## Gazebo Harmonic + ROS 2 Jazzy (WSL)

The distro `Ubuntu-24.04-ROS2` is imported from Ubuntu's official base image, so the
existing distros are untouched:

```powershell
$d = "$env:TEMP\noble"; mkdir $d -Force
$b = "https://cdimage.ubuntu.com/ubuntu-base/releases/24.04/release"
curl.exe -L -o "$d\base.tar.gz" "$b/ubuntu-base-24.04.5-base-amd64.tar.gz"   # check against $b/SHA256SUMS
wsl --import Ubuntu-24.04-ROS2 "$env:USERPROFILE\WSL\Ubuntu-24.04-ROS2" "$d\base.tar.gz" --version 2
wsl -d Ubuntu-24.04-ROS2 -u root -- bash /mnt/c/<path>/limo/wsl_bootstrap_root.sh <linux-user>
wsl --terminate Ubuntu-24.04-ROS2
wsl -d Ubuntu-24.04-ROS2 -- bash /mnt/c/<path>/limo/setup_wsl_jazzy.sh
```

The workspace `~/limo_ros2_ws` is built with `--symlink-install` from the repo's packages,
so edits to launch files, worlds, URDF and the bridge apply on the next start.

Start, watch, drive, stop (PowerShell, from `limo/`):

```powershell
$limo = "/mnt/c/Users/ehaye/Documents/Workspace/Active/cadence-test/limo"
Start-Process -WindowStyle Hidden wsl.exe -ArgumentList '-d','Ubuntu-24.04-ROS2','--','bash','-c',"`"bash $limo/run_sim_jazzy.sh gui:=true > ~/sim.log 2>&1`""
Start-Process py -ArgumentList '-m','limo_brain.viewer'           # native top-down viewer
py -m limo_brain.run --env bridge --minutes 15 --profile orin-nano-super --calibration rover.json
wsl -d Ubuntu-24.04-ROS2 -- bash $limo/stop_sim_jazzy.sh
```

The sim is ready in about 10 s and runs at real time. The default task is `--task goal` in
the open arena (`layout:=open`, walls only; `layout:=boxes` with `--layout boxes` adds five
boxes). `--task roam` is the earlier drive-without-bumping task.

Two ways to watch:

- **`py -m limo_brain.viewer`** is a native Windows window (tkinter). It shows the arena, the
  robot and its trail, the lidar points, the goal and its 0.3 m arrival zone, the path of the
  brain's chosen command, and live stats. It does not depend on WSLg, and works the same for
  the toy sim, Gazebo and the rover.
- **`gui:=true`** opens Gazebo's own window as a separate process (`gui_jazzy.sh`), so the
  simulation keeps running if the window freezes or closes; reopen it with `gui_jazzy.sh`.
  It renders on the GPU through D3D12. The controller draws the goal, the chosen command's
  path and the 8 sectors as Gazebo markers (`--no-viz` turns them off). If the window stays
  minimized or invisible, run `show_gazebo.ps1`.

Other launch options: `scan_hz:=10`, `body_mesh:=true`, `bridge:=false`. `--stuck-reset N`
and falls (IMU tilt over 45 degrees) put the robot back at its spawn pose through Gazebo's
`set_pose` (its WorldControl model reset did not move the robot). The bridge re-anchors
odometry at each reset, so reported poses stay in the arena frame. Walls are slippery so the
grippy tyres cannot climb them.

### WSL pitfalls found while setting this up

- **Invisible Gazebo window** (title starts with `[WARN:COPY MODE]`): a WSL 2.7.x bug,
  [microsoft/wslg#1456](https://github.com/microsoft/wslg/issues/1456), in which
  `/mnt/shared_memory` is missing and every WSLg window stays invisible. To fix it until the
  next WSL restart, restart the WSLg compositor, then start the sim again:
  `wsl -d Ubuntu-24.04-ROS2 --system -- sh -c 'kill $(pgrep -x weston)'`.
  `show_gazebo.ps1` detects this state. The good state shows
  `use_gfxredir = 1` in `/mnt/wslg/weston.log`.

- **Mirrored networking** (`networkingMode=mirrored` in `.wslconfig`): connecting to a
  *closed* port on `127.0.0.1` hangs for about 2 minutes instead of failing. This hits the
  ROS 2 CLI daemon (use `--no-daemon`, e.g. `ros2 topic list --no-daemon`) and ROS1's
  master check. Open ports and DDS work normally.
- **Robot spawn** reads the URDF as a string; waiting on `/robot_description` stalled for
  minutes.
- **Rendering.** The simulator renders the lidar in software (llvmpipe), at real time with
  the 240 degree scan. `SIM_GPU=1` renders it through WSL's D3D12 GPU driver, but Gazebo
  segfaulted inside `libd3d12core.so` after about 5 minutes. The window renders on the GPU
  (`GALLIUM_DRIVER=d3d12`, `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`,
  `LD_LIBRARY_PATH=/usr/lib/wsl/lib`). Check with `glxinfo -B`: the renderer should read
  `D3D12 (NVIDIA Quadro RTX 4000)`, not `llvmpipe`.
- **Gazebo's window under WSLg is unstable**
  ([gz-sim#3335](https://github.com/gazebosim/gz-sim/issues/3335): it crashes or fails to
  open on WSLg, no fix yet). Hence the separate window process and the native viewer.
  Avoid VcXsrv: it renders in software and is slower than WSLg.
- **Starting in the background:** use `Start-Process` as above, not a backgrounded
  `wsl.exe` from Git Bash.

## Real robot (later)

The rover's driver currently refuses to move the base, by design. When motion is allowed:
start the chassis driver with commands enabled and the lidar driver, then
`ros2 launch limo_cadence_bridge robot.launch.py` (speeds clamped to 0.3 m/s and 1.2 rad/s,
stop after 0.5 s without commands or on disconnect). Run the controller on the rover with
`--stuck-reset 0`, so a fall stops the run because nothing can respawn the robot. Lift the
wheels off the ground for the first test.

## Status (2026-10-15)

Toy sim, seed 0:

| Controller | Mean reward per tick | Bump share | Notes |
| --- | --- | --- | --- |
| `reflex` (hand-written) | 0.38 | 0.00 | The task is solvable |
| `random` | -0.22 | 0.27 | |
| Cadence, need 0.3 | -0.20 to -0.27 over 8000 ticks | 0.33 to 0.45 | Stays aroused and exploring |
| Cadence, need 0.3, rover calibration | -0.14 to -0.33 over 4000 ticks | 0.33 to 0.48 | Your run: 42 respawns, 9 overruns |
| Cadence, need 0 (founder default) | -0.30 | 0.00 | Settles on spinning in place, calm |

Cadence has not yet learned obstacle avoidance. The loop, sims, bridge and timing work end
to end; tuning is next (learning rates, credit across delay, observation, reward).

## Legacy: ROS1 sim for the original Jetson Nano LIMO

`ros/`, `fetch_limo_sim.sh`, `setup_wsl_noetic.sh`, `run_sim_wsl.sh` and `stop_sim_wsl.sh`
are the earlier ROS1 Noetic + Gazebo 11 setup (distro `Ubuntu-20.04-ROS`), built before the
rover turned out to run ROS 2 Jazzy. It works but does not match the rover. Remove the
distro with `wsl --unregister Ubuntu-20.04-ROS` if you no longer need it.
