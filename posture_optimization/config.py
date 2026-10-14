import json
from pathlib import Path
import numpy as np
from .geometry import transform, axis_frame


def load_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text(encoding="utf-8-sig"))
    c["robot"]["urdf"] = str((path.parent / c["robot"]["urdf"]).resolve())
    validate(c)
    return c


def validate(c):
    def positive(value, name):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    def integer(value, name):
        positive(value, name)
        if int(value) != value:
            raise ValueError(f"{name} must be an integer")
    if not Path(c["robot"]["urdf"]).is_file():
        raise ValueError("URDF file does not exist")
    transform(c["robot"]["base_pose_mm_deg"])
    transform(c["robot"]["tcp_pose_mm_deg"])
    rpy = np.asarray(c["orientation_rpy_deg"], float)
    if rpy.shape != (3,) or not np.isfinite(rpy).all():
        raise ValueError("orientation_rpy_deg requires three finite values")
    if c.get("reference_axis"):
        axis_frame(c["reference_axis"])
    for name in ("target", "initial"):
        region = c[name]
        if region["kind"] not in ("box", "sphere", "points", "cylinder", "half_cylinder", "half_annulus"):
            raise ValueError("Unsupported region kind")
        if region["kind"] != "points":
            positive(region["spacing_mm"], name+".spacing_mm")
        tolerance = region.get('boundary_tolerance_mm', 1e-7)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('boundary_tolerance_mm must be finite and nonnegative')
        if 'spacing_xyz_mm' in region:
            spacing = np.asarray(region['spacing_xyz_mm'], float)
            if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
                raise ValueError('spacing_xyz_mm requires three positive finite values')
        if region["kind"] == "box":
            lo, hi = np.asarray(region["min_mm"], float), np.asarray(region["max_mm"], float)
            if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo > hi):
                raise ValueError("Invalid box bounds")
        elif region["kind"] == "sphere":
            center = np.asarray(region["center_mm"], float)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError("Invalid sphere center")
            positive(region["radius_mm"], "radius_mm")
        elif region["kind"] == "points":
            points = np.asarray(region["points_mm"], float)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                raise ValueError("points_mm requires a nonempty finite Nx3 array")
        else:
            axis_frame(region.get("axis", c.get("reference_axis")))
            if region.get("side", "base") not in ("base", "positive", "negative"):
                raise ValueError("Unknown half-region side")
            normal = np.asarray(region.get("split_normal", [0, 1, 0]), float)
            if normal.shape != (3,) or not np.isfinite(normal).all():
                raise ValueError("Invalid split normal")
            positive(region["outer_radius_mm"], "outer_radius_mm")
            inner = region.get("inner_radius_mm", 0)
            if not 0 <= inner < region["outer_radius_mm"]:
                raise ValueError("Require 0 <= inner radius < outer radius")
            if region['kind'] == 'half_cylinder' and inner != 0:
                raise ValueError('half_cylinder requires zero inner radius')
    for key in ("count", "attempts", "ik_branches"):
        integer(c["candidates"][key], "candidates."+key)
    positive(c["candidates"]["dedup_deg"], "dedup_deg")
    for q in c["candidates"].get("joint_seeds_deg", []):
        if np.asarray(q).ndim != 1 or len(q) < 6 or not np.isfinite(q).all():
            raise ValueError("joint_seeds_deg requires at least six finite degree values per row")
    if not isinstance(c["seed"], int) or not 0 <= c["seed"] < 2**31:
        raise ValueError("seed must be an integer in [0, 2^31)")
    for key in ("cartesian_step_mm", "joint_step_deg", "max_joint_jump_deg", "position_tolerance_mm", "orientation_tolerance_deg", "characteristic_length_m"):
        positive(c["validation"][key], key)
    if not 0 < c["validation"]["inverse_condition_min"] < 1:
        raise ValueError("inverse_condition_min must lie in (0,1)")
    integer(c["optimization"]["k"], "k")
    positive(c["optimization"]["time_limit_s"], "time_limit_s")
    for key in ("de_maxiter", "de_popsize"):
        integer(c["optimization"][key], key)
    if c["optimization"]["method"] not in ("cp_sat", "differential_evolution", "both"):
        raise ValueError("Unknown optimization method")
    if not 0 <= c["validation"]["collision_margin_mm"] < 100:
        raise ValueError("Invalid collision margin")
    if not isinstance(c["validation"].get("collision_check", False), bool):
        raise ValueError("collision_check must be a boolean")
    if c["validation"].get("planner", "seeded_ik") not in ("seeded_ik", "descartes"):
        raise ValueError("planner must be seeded_ik or descartes")
