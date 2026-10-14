import numpy as np
import pytest
from posture_optimization.geometry import contains, sample_region, transform

AXIS = {"start_mm": [300, -600, 1000], "end_mm": [-300, -600, 1000]}


def test_base_facing_half_annulus_and_boundaries():
    region = {"kind": "half_annulus", "inner_radius_mm": 100, "outer_radius_mm": 250, "spacing_mm": 50}
    points = np.array([[0, -.45, 1], [0, -.75, 1], [0, -.55, 1], [.4, -.45, 1], [0, -.6, .75]])
    assert contains(points, region, AXIS, [0, 0, 0]).tolist() == [True, False, False, False, True]
    sample = sample_region(region, AXIS, [0, 0, 0])
    assert contains(sample, region, AXIS, [0, 0, 0]).all()
    assert np.all(sample[:, 1] >= -.6-1e-10)


def test_world_box_and_rotated_base_transform():
    region = {"kind": "box", "min_mm": [0, 0, 0], "max_mm": [100, 100, 100], "spacing_mm": 50}
    assert len(sample_region(region, AXIS, [0, 0, 0])) == 27
    t = transform([100, 200, 300, 0, 0, 90])
    assert np.allclose(t[:3, :3] @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(t[:3, 3], [.1, .2, .3])


def test_grid_budget():
    region = {"kind": "box", "min_mm": [0, 0, 0], "max_mm": [100, 100, 100], "spacing_mm": .01}
    with pytest.raises(ValueError, match="too large"):
        sample_region(region, AXIS, [0, 0, 0])


def test_half_cylinder_keeps_base_side_and_solid_center():
    axis = {'start_mm': [-2000, -400, 1000], 'end_mm': [2000, -400, 1000]}
    region = {'kind': 'half_cylinder', 'outer_radius_mm': 100, 'spacing_mm': 50,
              'split_normal': [0, 1, 0]}
    points = np.array([[0, -.4, 1], [0, -.3, 1], [0, -.5, 1],
                       [0, -.4, .9], [2.001, -.4, 1], [0, -.29, 1]])
    assert contains(points, region, axis, [-.59, 0, 0]).tolist() == [True, True, False, True, False, False]
    assert contains(points[:3], region, axis, [-.59, -1, 0]).tolist() == [True, False, True]
    sampled = sample_region(region, axis, [-.59, 0, 0])
    assert len(sampled) == 729
    assert np.all(sampled[:, 1] >= -.4-1e-10)
    assert np.allclose(sampled.min(axis=0), [-2, -.4, .9])
    assert np.allclose(sampled.max(axis=0), [2, -.3, 1.1])


def test_narrow_half_cylinder_uses_independent_axial_spacing():
    axis = {'start_mm': [-2000, -400, 1000], 'end_mm': [2000, -400, 1000]}
    region = {'kind': 'half_cylinder', 'outer_radius_mm': 20,
              'spacing_mm': 10, 'spacing_xyz_mm': [50, 10, 10]}
    points = sample_region(region, axis, [-.59, 0, 0])
    assert len(points) == 729
    assert len(np.unique(points[:, 0])) == 81
    assert np.all(points[:, 1] >= -.4-1e-10)


def test_ik_roundoff_does_not_exclude_half_annulus_boundary():
    axis = {'start_mm': [-2000, -600, 1000], 'end_mm': [2000, -600, 1000]}
    region = {'kind': 'half_annulus', 'inner_radius_mm': 100, 'outer_radius_mm': 250,
              'spacing_mm': 50, 'boundary_tolerance_mm': .001}
    points = [[-.25, -.349999978699007, .9999998496393663],
              [-.25, -.349, 1.0]]
    assert contains(points, region, axis, [-.59, 0, 0]).tolist() == [True, False]
    plane = [[-.25, -.60000005, 1.15], [-.25, -.600002, 1.15]]
    assert contains(plane, region, axis, [-.59, 0, 0]).tolist() == [True, False]
