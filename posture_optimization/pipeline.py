import json
import platform
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .backend import TesseractBackend
from .geometry import contains, sample_region
from .movl import MovLValidator
from .optimize import coverage, optimize


def candidates(backend, validator, config, rotation, log=print):
    options = config["candidates"]
    rng = np.random.default_rng(config["seed"])
    pool = sample_region(config["initial"], config.get("reference_axis"), backend.base_pose[:3, 3])
    if not len(pool):
        raise ValueError("Initial region sampling is empty; decrease spacing_mm")
    result, failures = [], Counter()

    def accept(q):
        if q is None:
            failures["ik_failed"] += 1
            return
        if options.get("canonical_angles", False):
            canonical = (q+np.pi) % (2*np.pi)-np.pi
            if np.all(canonical >= backend.limits[:, 0]) and np.all(canonical <= backend.limits[:, 1]):
                q = canonical
        pose = backend.fk(q)
        angle = Rotation.from_matrix(rotation.T @ pose[:3, :3]).magnitude()
        if angle > np.deg2rad(config["validation"]["orientation_tolerance_deg"]):
            failures["orientation_error"] += 1
            return
        if not contains(pose[:3, 3], config["initial"], config.get("reference_axis"), backend.base_pose[:3, 3])[0]:
            failures["outside_initial_region"] += 1
            return
        reason, _ = validator.state_reason(q)
        if reason != "ok":
            failures[reason] += 1
            return
        if any(np.max(abs(q-existing)) < np.deg2rad(options["dedup_deg"]) for existing in result):
            failures["duplicate"] += 1
            return
        result.append(q)

    for joint_seed in options.get("joint_seeds_deg", []):
        q = np.deg2rad(joint_seed)
        if q.shape != (len(backend.joint_names),):
            raise ValueError("Joint seed dimension must match selected URDF chain")
        reason, _ = validator.state_reason(q)
        if reason != "ok":
            failures[reason] += 1
            continue
        accept(q)
        if len(result) >= options["count"]:
            break
    # Sample joint space for seeds, then solve fixed-orientation IK at sampled XYZ.
    order = rng.permutation(len(pool))
    for attempt in range(options["attempts"]):
        if len(result) >= options["count"]:
            break
        position = pool[order[attempt % len(order)]]
        for _ in range(options["ik_branches"]):
            seed = rng.uniform(backend.limits[:, 0], backend.limits[:, 1])
            q = backend.ik(position, rotation, seed)
            if q is not None and np.linalg.norm(backend.fk(q)[:3, 3]-position)*1000 > config["validation"]["position_tolerance_mm"]:
                failures["ik_residual"] += 1
            else:
                accept(q)
            if len(result) >= options["count"]:
                break
        if (attempt+1) % 50 == 0:
            log(f"Candidate search {attempt+1}/{options['attempts']}: {len(result)} accepted", flush=True)
    return np.asarray(result).reshape(-1, len(backend.joint_names)), dict(failures)


def run(config, output, show=False, log=print):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    backend = TesseractBackend(config)
    try:
        rotation = Rotation.from_euler("xyz", config["orientation_rpy_deg"], degrees=True).as_matrix()
        validator = MovLValidator(backend, config["validation"])
        points = sample_region(config["target"], config.get("reference_axis"), backend.base_pose[:3, 3])
        if not len(points):
            raise ValueError("Target region sampling is empty; decrease spacing_mm")
        qs, rejected = candidates(backend, validator, config, rotation, log)
        matrix = np.zeros((len(qs), len(points)), dtype=bool)
        reasons = np.full(matrix.shape, "untested", dtype="U32")
        stats = []
        for i, q in enumerate(qs):
            reports = []
            for j, point in enumerate(points):
                # Each test starts at the exact same initial joint vector.
                validation = validator.plan(q.copy(), point, rotation)
                matrix[i, j] = validation.success
                reasons[i, j] = validation.reason
                reports.append(validation.report())
            stats.append(reports)
            log(f"MovL {i+1}/{len(qs)}: {matrix[i].sum()}/{len(points)} reachable", flush=True)
        selection = optimize(matrix, config["optimization"], config["seed"])
        selected = selection["indices"]
        covered = coverage(matrix, selected)
        packages = {name: version(name) for name in ("tesseract-robotics-nanobind", "numpy", "scipy", "ortools", "matplotlib")}
        result = {"schema_version": 1, "seed": config["seed"], "backend": "tesseract_kdl_bullet",
                  "robot_urdf": config["robot"]["urdf"],
                  "base_pose_mm_deg": config["robot"]["base_pose_mm_deg"],
                  "tcp_pose_mm_deg": config["robot"]["tcp_pose_mm_deg"],
                  "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
                  "joint_names": backend.joint_names, "target_count": len(points), "candidate_count": len(qs),
                  "candidate_rejections": rejected, "failure_counts": dict(Counter(reasons.ravel().tolist())),
                  "selection": selection, "selected_joints_deg": np.rad2deg(qs[selected]).tolist(),
                  "per_candidate_coverage": matrix.mean(axis=1).tolist(), "seconds": time.perf_counter()-started,
                  "collision_check": config["validation"].get("collision_check", False),
                  "validation_scope": "Discrete Cartesian points and dense joint interpolation; optional convex-hull contact checks"}
        # Reusing an output directory must not retain trajectories for starts
        # selected by an earlier configuration/run.
        for pattern in ("candidate_*_reachable_mm.csv", "trajectory_*.npz"):
            for previous in output.glob(pattern):
                previous.unlink()
        np.savez_compressed(output/"coverage.npz", A=matrix, targets_m=points, candidates_rad=qs,
                            reasons=reasons, selected_indices=np.asarray(selected, int), covered=covered)
        np.savetxt(output/"targets_mm.csv", points*1000, delimiter=",", header="x_mm,y_mm,z_mm", comments="")
        np.savetxt(output/"candidates_deg.csv", np.rad2deg(qs), delimiter=",", header=",".join(backend.joint_names), comments="")
        np.savetxt(output/"coverage_matrix.csv", matrix.astype(int), delimiter=",", fmt="%d")
        np.savetxt(output/"covered_points_mm.csv", points[covered]*1000, delimiter=",", header="x_mm,y_mm,z_mm", comments="")
        for index in selected:
            np.savetxt(output/f"candidate_{index}_reachable_mm.csv", points[matrix[index]]*1000,
                       delimiter=",", header="x_mm,y_mm,z_mm", comments="")
        for filename, data in (("result.json", result), ("resolved_config.json", config), ("validation.json", stats)):
            (output/filename).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        # Save one successful trajectory for every selected start (if one exists).
        for index in selected:
            reachable = np.flatnonzero(matrix[index])
            if len(reachable):
                point_index = int(reachable[0])
                trajectory = validator.plan(qs[index].copy(), points[point_index], rotation)
                if not trajectory.success:
                    raise RuntimeError("Selected trajectory could not be reproduced")
                np.savez_compressed(output/f"trajectory_{index}.npz", joints_rad=trajectory.trajectory,
                                    target_m=points[point_index], tcp_m=np.asarray([backend.fk(q)[:3, 3] for q in trajectory.trajectory]))
        from .visualize import plot
        plot(points, qs, matrix, selected, backend, config, output/"coverage.png", show)
        log(json.dumps(selection, ensure_ascii=False, indent=2))
        return result
    finally:
        backend.close()
