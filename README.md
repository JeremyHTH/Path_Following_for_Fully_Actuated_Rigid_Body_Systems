Path Following for Fully Actuated Rigid-Body Systems
====================================================

This project provides a self‑contained Python package for generating smooth
position/orientation paths, building intrinsic frames, and simulating a fully
actuated manipulator that tracks those paths. Two demo scripts illustrate the
workflow: one focused on path utilities and another that runs a closed-loop robot
simulation with logging and visualisation.


Package Overview (`Path_Following_Package`)
-------------------------------------------

The package exposes the following key components:

| Module | Description |
|--------|-------------|
| `Path_Generation.py` | Builds $C^n$ splines through Cartesian waypoints and associated orientations, handling both open and closed paths. |
| `Frame_Path.py` | Extends the generated path with arc-length parametrisation and parallel-transport frames for curvature-aware control. |
| `Control.py` | Implements the virtual task-space controller that tracks a reference path with orientation regulation. |
| `Robot_Dynamic.py` | Provides a serial manipulator model (MDH/DH) with kinematics, Jacobians, and dynamics utilities. |
| `util.py` | Helper functions for quaternion math, $\mathfrak{so}(3)$ logarithms, and wrapped arc-length errors. |

Typical workflow:

```python
from Path_Following_Package import Frame_Path, Virtal_Task_Space_Control_Module, Robot

frame = Frame_Path(points, rotations, Is_loop=False, n_continuity=4)
controller = Virtal_Task_Space_Control_Module()
robot = Robot(Number_of_joint, ALPHA, A, D, OFFSET, MASS, COM, INERTIA, Fv, Fs, Gravity)

v, diagnostics = controller.Get_Control_Input(robot, frame, q, qd, qdd, reference_signal)
```

Inputs & Requirements
---------------------

To run the complete algorithm you must supply:

* **Path waypoints** – a list/array of 3D points describing the desired curve.
* **Orientation samples** – one rotation matrix per waypoint establishing the
  frame the end-effector should align with along the path.
* **Robot parameters** – Denavit–Hartenberg (DH or MDH) parameters for the target
  manipulator, including link dimensions, joint types, inertial properties, and
  optional end-effector offsets.

These inputs feed the path generator (`Frame_Path`) and the robot model
(`Robot`), which are then connected by the virtual task-space controller.


Demo Scripts
------------

### 1. `Package_Demo.py`
Demonstrates the fundamental path utilities without the closed-loop robot:

* Generates a looping 3D path with smoothly varying orientations.
* Uses `Frame_Path` to visualise the curve, the moving frames, and the closest-point
  projection for several query positions.
* Instantiates the 7-DoF KUKA iiwa model to print all forward-kinematics frames,
  the Jacobian, and its time derivative.

**Run the demo**
```bash
python Package_Demo.py
```
Plots open in interactive windows; nothing is saved to disk by default.


### 2. `Demo_on_simple_path.py`
Runs a full simulation where the KUKA iiwa follows a chosen path while logging
metrics and exporting plots to the `Demo_Image/` directory.

Key features:

* Three preconfigured path options selected via `Current_Path` (`"Circular"`,
  `"Linear"`, `"Random_smooth"`). Each branch builds the waypoint and orientation lists
  before constructing the `Frame_Path`.
* `Reference_signal` provides speed profiles for open vs. closed paths.
* The virtual task-space controller logs task variables, joint states, orientation
  errors, and controller outputs; plots are saved with a prefix (`PREFIX_NAME`)
  reflecting the chosen path.

**Run the demo**
```bash
python Demo_on_simple_path.py
```
Outputs include PNG figures (stored in `Demo_Image/`) and an Excel export of the
task-space Jacobian history.

### 3. `Demo_on_Real_time_swap_path.py`
Illustrates online re-targeting: the robot starts on one closed path and, mid-simulation,
switches to a different closed path without resetting the controller state.

Key features:

* Precomputes two looping paths (`Frame_1` and `Frame_2`) with distinct geometry and orientation profiles.
* Feeds the same manipulator and controller but swaps the active frame in real time, demonstrating that the Frenet-frame feedback and closest-point search handle sudden path changes.
* Generates the same suite of plots so the transient behaviour during the path swap can be analysed.

**Run the demo**
```bash
python Demo_on_Real_time_swap_path.py
```
Plots are saved to `Demo_Image/` with the `Mix_Path_` prefix.

Reference
---------

- G. Loianno, G. A. Muñoz, and V. Kumar, "Combined path following and compliance control for fully actuated rigid body systems in 3-D space," *IEEE Robotics and Automation Letters*, vol. 3, no. 3, pp. 2310–2317, 2018.
- A. Akhtar and S. L. Waslander, "Controller class for rigid body tracking on SO(3)," *IEEE Transactions on Control Systems Technology*, vol. 27, no. 1, pp. 263–270, 2019.

Sample Results
--------------

The `Demo_Image/` folder contains plots generated by `Demo_on_simple_path.py`.
Selected outputs for each path option are shown below:

### Circular Path
![Circular Path – 3D trajectory](Demo_Image/Circular_Path_end_effector_3d.png)
![Circular Path – Cartesian tracking](Demo_Image/Circular_Path_end_effector.png)
![Circular Path – η/ξ task variables](Demo_Image/Circular_Path_Eta_Xi.png)
![Circular Path – Quaternion convergence](Demo_Image/Circular_Path_Quaternion.png)

### Linear Path
![Linear Path – 3D trajectory](Demo_Image/Linear_Path_end_effector_3d.png)
![Linear Path – Cartesian tracking](Demo_Image/Linear_Path_end_effector.png)
![Linear Path – η/ξ task variables](Demo_Image/Linear_Path_Eta_Xi.png)
![Linear Path – Quaternion convergence](Demo_Image/Linear_Path_Quaternion.png)

### Random Smooth Path
![Random Smooth Path – 3D trajectory](Demo_Image/Random_Smooth_Path_end_effector_3d.png)
![Random Smooth Path – Cartesian tracking](Demo_Image/Random_Smooth_Path_end_effector.png)
![Random Smooth Path – η/ξ task variables](Demo_Image/Random_Smooth_Path_Eta_Xi.png)
![Random Smooth Path – Quaternion convergence](Demo_Image/Random_Smooth_Path_Quaternion.png)

### Real-Time Path Swap
![Real-Time Path Swap – 3D trajectory](Demo_Image/Mix_Path_end_effector_3d.png)
![Real-Time Path Swap – Cartesian tracking](Demo_Image/Mix_Path_end_effector.png)
![Real-Time Path Swap – η/ξ task variables](Demo_Image/Mix_Path_Eta_Xi.png)
![Real-Time Path Swap – Quaternion convergence](Demo_Image/Mix_Path_Quaternion.png)


Prerequisites & Setup
---------------------

* Python 3.8+
* NumPy, SciPy, Matplotlib, SymPy, pandas, tqdm

Install requirements (if not already available):
```bash
pip install numpy scipy matplotlib sympy pandas tqdm
```


Repository Layout
-----------------

```
Path_Following_for_Fully_Actuated_Rigid_Body_Systems/
├── Path_Following_Package/      # Core library
├── Package_Demo.py              # Path and frame utility showcase
├── Demo_on_simple_path.py       # Full path-following simulation
├── Demo_Image/                  # Generated figures (created on demand)
└── README.md
```


Getting Started
---------------

1. Review `Demo_on_simple_path.py` to choose a path type and adjust controller gains,
   simulation length, or logging settings as needed.
2. Run the demo to produce figures and inspect the generated logs.
3. Use `Package_Demo.py` as a quick sandbox for experimenting with new point sets or
   robot parameters before integrating them into the closed-loop example.

Precautions
-----------

For closed paths, the parallel-transport frame at the start point and the end point may not be identical. This can result in a brief instability when the reference signal wraps from the end of the path back to the beginning.

For open paths, it can take longer for the robot to reacquire the trajectory if it does not start on the path. The control logic first seeks the closest point on the curve rather than the geodesically shortest path to the desired point, so transient detours may occur before convergence.
