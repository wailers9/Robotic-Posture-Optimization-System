# Robotic Posture Optimization System

A configurable framework for robot posture optimization, Cartesian workspace evaluation, multi-scenario coverage analysis, and singularity monitoring.

The system searches for starting joint configurations from which a robot can execute straight Cartesian motions to a set of target points while maintaining a specified TCP orientation. Robot geometry, mounting pose, TCP calibration, initial region, target region, and validation settings are supplied through JSON configuration. A bundled robot provides a runnable demonstration; application logic does not depend on its name, link naming convention, arm side, a particular workpiece, or a fixed number of tools.

## Capabilities

- Load a robot from URDF, resolve local or package mesh paths, and select a configurable tip link.
- Apply arbitrary robot base and TCP transforms.
- Support bounded revolute serial chains with six or more active joints, including redundant chains.
- Sample boxes, spheres, finite cylinders, half cylinders, half annuli, and explicit point sets.
- Generate multiple IK branches and reject invalid, singular, out-of-region, or duplicate starts.
- Evaluate each candidate independently against every target using Cartesian interpolation and seeded IK, or an optional Descartes planner.
- Verify joint limits, joint jumps, Cartesian line deviation, orientation error, endpoint accuracy, motion progress, and normalized Jacobian conditioning along interpolated paths.
- Select one or several postures with SciPy differential evolution and/or OR-Tools CP-SAT maximum coverage optimization.
- Compare task scenarios, identify the best scenario/posture pair, and select a shared posture or posture library across scenarios.
- Display TCP position, six Jacobian singular values, and inverse condition number in a local monitoring webpage.
- Export machine-readable configuration snapshots, coverage matrices, failure reports, plots, and example trajectories.

Collision checking is disabled by default. Optional collision geometry and Bullet checks remain available through configuration.

## Installation

Python 3.10 or newer is required; this version was verified with Python 3.11 on Windows. Availability of the pinned Tesseract wheel depends on the platform and Python version.

```powershell
git clone https://github.com/wailers9/Robotic-Posture-Optimization-System.git
cd Robotic-Posture-Optimization-System
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install pytest
```

On systems with a compatible Tesseract wheel:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install pytest
```

Examples below assume that the virtual environment is activated. Otherwise replace `python` with its virtual-environment executable. The installed command `posture-opt` exposes the same interface as `python -m posture_optimization`.

## Quick start

```bash
python -m posture_optimization doctor
python -m posture_optimization run --config config/default.json --output outputs/demo
python -m posture_optimization search-scenarios --config config/scenarios.json --output outputs/scenarios
python -m posture_optimization monitor --config config/default.json --port 8790
```

Open `http://127.0.0.1:8790` for the monitoring page. The bundled demonstration uses an identity mounting transform, a simple TCP offset, and box-shaped regions. These are editable example settings, not assumptions of the optimizer.

`doctor` finds a start within the configured initial region and checks a short 20 mm world-X move. Its report includes both seeded-IK and Descartes results. It exits successfully when the seeded-IK diagnostic passes; the optional Descartes report can still reject a branch transition, for example with `joint_jump`.

## Coordinate conventions

| Quantity | JSON / CLI | Internal representation |
| --- | --- | --- |
| Position / distance | millimetres | metres |
| Joint angles | degrees | radians |
| Pose | `[x, y, z, roll, pitch, yaw]` | 4 × 4 transform |
| Orientation | `[roll, pitch, yaw]`, degrees | 3 × 3 rotation |
| Workspace | world coordinates | world coordinates |
| Base pose | world to original URDF root | fixed transform |
| TCP pose | configured tip link to TCP | fixed transform |

Rotations use `Rotation.from_euler("xyz", ..., degrees=True)`, equivalent to `Rz(yaw) @ Ry(pitch) @ Rx(roll)`. The full TCP transform is `world_T_root @ root_T_tip(q) @ tip_T_tcp`. Changing TCP translation or orientation changes IK, workspace eligibility, Jacobian conditioning, and coverage.

The desired TCP orientation is fixed during each scenario's motions. Variable-orientation trajectories are outside the current task model.

## Configure a robot and task

Start with [config/default.json](config/default.json). All listed sections are required except `reference_axis` and `obstacles`.

```json
{
  "seed": 42,
  "robot": {
    "urdf": "../models/cr7.urdf",
    "tip_link": "Link6",
    "base_pose_mm_deg": [0, 0, 0, 0, 0, 0],
    "tcp_pose_mm_deg": [0, 0, 100, 0, 0, 0],
    "exclude_links": [],
    "allowed_collision_pairs": [],
    "package_paths": {}
  },
  "orientation_rpy_deg": [0, 0, 0],
  "initial": {
    "kind": "box", "min_mm": [-100, -500, 700],
    "max_mm": [100, -300, 900], "spacing_mm": 100,
    "boundary_tolerance_mm": 0.001
  },
  "target": {
    "kind": "box", "min_mm": [-200, -500, 700],
    "max_mm": [200, -300, 900], "spacing_mm": 100,
    "boundary_tolerance_mm": 0.001
  },
  "candidates": {
    "count": 6, "attempts": 150, "ik_branches": 4,
    "joint_seeds_deg": [], "dedup_deg": 3
  },
  "validation": {
    "collision_check": false, "planner": "seeded_ik",
    "cartesian_step_mm": 10, "joint_step_deg": 0.5,
    "max_joint_jump_deg": 8, "position_tolerance_mm": 0.5,
    "orientation_tolerance_deg": 0.25,
    "inverse_condition_min": 0.06, "characteristic_length_m": 0.5,
    "collision_margin_mm": 2
  },
  "optimization": {
    "k": 2, "method": "cp_sat", "time_limit_s": 10,
    "de_maxiter": 100, "de_popsize": 15
  },
  "obstacles": []
}
```

URDF paths resolve relative to the configuration file. Mesh paths resolve relative to the URDF. For `package://my_robot/meshes/link.stl`, set `robot.package_paths.my_robot` to that package's directory, relative to the URDF or absolute. The selected chain must contain bounded `revolute` and `fixed` joints only. Prismatic and unbounded continuous joints are rejected because the current public interface uses joint degrees and literal finite angular bounds.

Replace the example URDF and tip link to integrate another robot. Review its link origins, axes, limits, and mesh scale. A six-or-more-joint chain does not automatically guarantee full Cartesian rank; the validator checks the six-row task Jacobian. No controller SDK or hardware connection is required.

`joint_seeds_deg` contains optional starting postures, one value per active joint in the backend's reported order. They are validated before acceptance. `canonical_angles: true` optionally maps equivalent angles into `[-180, 180)` when those values are within the URDF limits. Otherwise literal angle branches are preserved; trajectories are never wrapped to hide a jump.

## Workspace shapes

Every region uses world coordinates. Except explicit point sets, provide `spacing_mm`; optional `spacing_xyz_mm: [sx, sy, sz]` replaces the isotropic spacing. Samples are filtered from a world-aligned grid. The upper grid boundary is included only when it lands on the spacing. A two-million-point budget limits the pre-filter grid.

| `kind` | Required shape fields |
| --- | --- |
| `box` | `min_mm`, `max_mm` |
| `sphere` | `center_mm`, `radius_mm` |
| `points` | nonempty `points_mm: [[x,y,z], ...]`; duplicate coordinates are removed |
| `cylinder` | `outer_radius_mm`, optional `inner_radius_mm`, `axis` |
| `half_cylinder` | `outer_radius_mm`, `axis`, optional `split_normal` and `side`; inner radius is zero |
| `half_annulus` | `outer_radius_mm`, `inner_radius_mm`, `axis`, optional `split_normal` and `side` |

Example axial region:

```json
{
  "kind": "half_cylinder",
  "axis": {"start_mm": [-300, -250, 600], "end_mm": [300, -250, 600]},
  "outer_radius_mm": 80,
  "split_normal": [0, 1, 0],
  "side": "positive",
  "spacing_mm": 20,
  "boundary_tolerance_mm": 0.001
}
```

Each axial region can have its own axis. Alternatively define top-level `reference_axis` and omit the region's `axis`. A reference axis alone does not create a physical obstacle. `split_normal` is projected perpendicular to the axis and defines a plane containing that axis. `side` is `positive`, `negative`, or `base` (the default). The base option chooses the side containing the robot base and rejects a base exactly in the split plane.

`boundary_tolerance_mm` is a numerical membership tolerance. It is separate from path position tolerance and does not create new grid samples.

## Optimization and validation

1. Sample the initial region, solve multiple seeded IK branches, and validate accepted starting states.
2. For each start/target pair, begin at the same exact joint vector. Generate a straight Cartesian path with fixed orientation and solve intermediate states.
3. Independently check dense joint-interpolated subsegments against the Cartesian line, orientation, joint bounds, jump threshold, endpoint, and motion progress.
4. At each checked state, normalize the translational Jacobian rows by `characteristic_length_m`, compute SVD, and require `sigma_min / sigma_max >= inverse_condition_min`.
5. Build a Boolean matrix `A[i,j]` indicating a successfully validated path.
6. Select `k` distinct starts maximizing the union of covered target samples.

`cp_sat` models selection and target coverage as Boolean variables. `differential_evolution` searches integer candidate indices, penalizes duplicate selections, and retains a greedy incumbent. `both` runs DE followed by CP-SAT. Requested `k` is reduced to the available candidate count when necessary. Empty candidate pools produce `NO_CANDIDATES` and zero coverage.

An `OPTIMAL` result proves the selection objective over the generated finite candidate pool and sampled targets. It does not prove a global optimum over continuous joint space, an unsampled continuous workspace, or unlisted scenarios. Candidate generation, grid density, model calibration, and validation thresholds affect results. Collision checks, when enabled, use mesh convex hulls and Bullet discrete/continuous contact checks. These geometric checks do not perform timing, acceleration, torque, or controller-specific execution analysis.

## Multi-scenario search

[config/scenarios.json](config/scenarios.json) demonstrates a nominal task, a translated workspace, and an alternate TCP. Supply any finite list of parameter alternatives:

```json
{
  "base_config": "default.json",
  "scenarios": [
    {"name": "nominal", "weight": 1, "overrides": {}},
    {"name": "alternate_mount", "weight": 2,
     "overrides": {"robot": {"base_pose_mm_deg": [50, 0, 0, 0, 0, 0]}}}
  ]
}
```

Overrides merge recursively with the base configuration; arrays replace entire arrays. Generate larger parameter searches by writing a scenario entry for each parameter combination. `base_config` and overridden relative URDF paths resolve relative to the manifest; the base configuration's resolved URDF is retained when not overridden.

All scenarios must use the same URDF, ordered joint names, and joint limits. Base transforms, TCPs, desired orientations, regions, obstacles, and validation parameters may vary. Joint posture candidates are pooled across scenarios and rechecked for eligibility in each scenario. An ineligible posture contributes zero coverage for that scenario.

The output provides:

- `best_scenario`: the highest single-start coverage scenario, its configuration index/name, joints, and world TCP pose.
- `shared_posture`: one joint posture maximizing weighted mean scenario coverage.
- `posture_library`: up to `k` shared starts maximizing union coverage.
- `per_scenario`: each scenario's coverage and selected-start eligibility count.

The aggregate objective is `sum(weight_s * covered_s / target_count_s) / sum(weight_s)`. Equal weights give equal influence to scenarios even when target counts differ. Rational weights are converted to exact integer coefficients for CP-SAT; oversized coefficients are rejected rather than silently rounded. Scenarios enumerate alternatives; no workpiece type or search plane is embedded in the solver.

## Singularity monitoring

```bash
python -m posture_optimization monitor --config config/default.json --host 127.0.0.1 --port 8790
```

The webpage builds joint inputs from the configured URDF chain, displays world TCP coordinates, plots the inverse condition number, and shows the six normalized Jacobian singular values. Enter joints manually or feed measured data using HTTP:

```bash
curl -X POST http://127.0.0.1:8790/api/metrics \
  -H "Content-Type: application/json" \
  -d '{"joints_deg":[0,20,-40,0,60,0]}'
```

```powershell
$body = @{joints_deg = @(0,20,-40,0,60,0)} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8790/api/metrics -Method Post -ContentType 'application/json' -Body $body
```

`GET /api/meta` reports joint order, limits, characteristic length, and threshold. `GET /api/latest` returns the most recent metrics, or `null` before the first submission. Invalid vectors and out-of-limit inputs return HTTP 400. The webpage polls the latest sample once per second; its trend records changed samples. An external adapter can read any controller's measured joints and post them in the reported order. The server computes model-based metrics and sends no robot commands.

## Outputs and reproducibility

| Single-task artifact | Contents |
| --- | --- |
| `result.json` | selected joints, objective/status, failures, runtime, package versions |
| `resolved_config.json` | exact configuration with resolved model path |
| `coverage.npz` | Boolean matrix, targets in metres, candidates in radians, failure reasons |
| `coverage_matrix.csv` | candidate rows × target columns |
| `targets_mm.csv`, `candidates_deg.csv` | sampled targets and candidate joint vectors |
| `covered_points_mm.csv` | selected union of targets |
| `candidate_*_reachable_mm.csv` | coverage of each selected start |
| `validation.json` | per-path validation reports |
| `trajectory_*.npz` | one reproduced successful trajectory per selected start with coverage |
| `coverage.png` | world workspace and selected robot skeletons |

Scenario search additionally writes `combined.npz`, `result.json`, and numbered `scenario_XXX` folders containing resolved configurations, eligibility arrays, and coverage matrices. Array column order is defined by the stored target arrays. A successful path to every covered point is evaluated, but only representative trajectories are retained by the single-task pipeline; scenario output stores feasibility matrices rather than all trajectories.

To independently validate a particular move:

```bash
python -m posture_optimization validate --config config/default.json \
  --joints-deg 0 20 -40 0 60 0 --target-mm 100 -400 800 \
  --output outputs/validation
```

Use a selected vector from `result.json` for a known candidate. A rejected move writes a report and exits with code 2. Configuration/backend errors exit with code 1. The seed controls candidate sampling and optimizers; results also depend on library versions and platform.

## Project structure

```text
config/                    Runnable task and scenario manifests
models/                    Demonstration URDF, meshes, model license
posture_optimization/
  backend.py               Tesseract KDL/Bullet/Descartes adapter
  config.py                JSON loading and validation
  geometry.py              Region membership and sampling
  movl.py                  Cartesian planning and independent verification
  optimize.py              Weighted maximum coverage solvers
  pipeline.py              Single-task search and exports
  scenarios.py             Scenario pooling, eligibility and shared selection
  monitor.py, monitor.html HTTP metrics API and monitoring page
  visualize.py             World-coordinate coverage plot
  cli.py                   Command-line entry points
tests/                     Geometry, path, optimizer, backend and API tests
```

Installable wheels include demonstration configurations, model assets, and the webpage. `--config` can point to an external task anywhere; outputs default to the current directory's `outputs/`.

## Tests

```bash
python -m pytest -q
```

The suite verifies geometry and membership tolerances, exact weighted coverage against exhaustive search, IK/FK consistency, finite-difference Jacobians, interpolated path rejection, redundant-joint dimensions, scenario normalization, and HTTP monitor responses. Tesseract integration tests use the included demonstration model and run without hardware.

## License and model provenance

Project code is distributed under [MIT](LICENSE). The bundled model has its own retained [MIT license](models/LICENSE); provenance and model modifications are described in [models/README.md](models/README.md). The example's application-specific tool geometry has been removed. TCP calibration is supplied exclusively by each task configuration.
