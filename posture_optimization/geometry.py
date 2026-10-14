"""World-frame sampling, independent of the robot backend."""
import numpy as np
from scipy.spatial.transform import Rotation


def transform(pose_mm_deg):
    pose = np.asarray(pose_mm_deg, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Pose requires six finite mm/degree values")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    t[:3, 3] = pose[:3] / 1000
    return t


def axis_frame(reference_axis):
    if reference_axis is None:
        raise ValueError("Axial regions require region.axis or reference_axis")
    a, b = np.asarray(reference_axis["start_mm"], float)/1000, np.asarray(reference_axis["end_mm"], float)/1000
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
        raise ValueError("Axis endpoints require three finite mm coordinates")
    length = np.linalg.norm(b-a)
    if length <= 0:
        raise ValueError("Axis endpoints must differ")
    axis = (b-a)/length
    return a, axis, length


def contains(points, region, reference_axis=None, base_position=(0, 0, 0)):
    p = np.atleast_2d(points)
    tolerance = region.get('boundary_tolerance_mm', 1e-7)/1000
    if region["kind"] == "box":
        return np.all((p >= np.asarray(region["min_mm"])/1000-tolerance) &
                      (p <= np.asarray(region["max_mm"])/1000+tolerance), axis=1)
    if region["kind"] == "sphere":
        return np.linalg.norm(p-np.asarray(region["center_mm"])/1000, axis=1) <= region["radius_mm"]/1000+tolerance
    if region["kind"] == "points":
        reference = np.asarray(region["points_mm"])/1000
        return np.array([np.any(np.linalg.norm(reference-point, axis=1) <= tolerance+1e-12) for point in p])
    a, axis, length = axis_frame(region.get("axis", reference_axis))
    delta = p-a
    axial = delta @ axis
    radial = delta-axial[:, None]*axis
    r = np.linalg.norm(radial, axis=1)
    mask = (axial >= -tolerance) & (axial <= length+tolerance)
    mask &= (r <= region["outer_radius_mm"]/1000+tolerance) & (r >= region.get("inner_radius_mm", 0)/1000-tolerance)
    if region["kind"] in ("half_annulus", "half_cylinder"):
        # Split by a configurable plane containing the region axis.
        normal = np.asarray(region.get("split_normal", [0, 1, 0]), float).copy()
        normal -= axis*(normal @ axis)
        if np.linalg.norm(normal) < 1e-8:
            raise ValueError("Split normal must not be parallel to axis")
        normal /= np.linalg.norm(normal)
        side = np.dot(np.asarray(base_position)-(a+axis*length/2), normal)
        side = {"positive": 1, "negative": -1}.get(region.get("side", "base"), side)
        if abs(side) < 1e-10:
            raise ValueError("Base lies in the half-region split plane")
        mask &= (radial @ normal)*np.sign(side) >= -tolerance
    return mask


def sample_region(region, reference_axis=None, base_position=(0, 0, 0), max_points=2_000_000):
    if region["kind"] == "points":
        points = np.unique(np.asarray(region["points_mm"], float), axis=0)/1000
        if len(points) > max_points:
            raise ValueError("Too many points")
        return points
    step = np.asarray(region.get('spacing_xyz_mm', [region['spacing_mm']]*3), float)/1000
    if region["kind"] == "box":
        lo, hi = np.asarray(region["min_mm"])/1000, np.asarray(region["max_mm"])/1000
    elif region["kind"] == "sphere":
        center, radius = np.asarray(region["center_mm"])/1000, region["radius_mm"]/1000
        lo, hi = center-radius, center+radius
    else:
        a, axis, length = axis_frame(region.get("axis", reference_axis))
        b = a+axis*length
        radius = region["outer_radius_mm"]/1000
        extent = radius*np.sqrt(np.maximum(0, 1-axis**2))
        lo, hi = np.minimum(a, b)-extent, np.maximum(a, b)+extent
    sizes = np.floor((hi-lo)/step+1e-8)+1
    if np.prod(sizes) > max_points:
        raise ValueError("Sampling grid too large; increase spacing_mm")
    axes = [np.arange(low, high+spacing*1e-8, spacing) for low, high, spacing in zip(lo, hi, step)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return points[contains(points, region, reference_axis, base_position)]
