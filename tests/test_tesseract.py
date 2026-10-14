from pathlib import Path
import numpy as np
import pytest
from posture_optimization.backend import TesseractBackend
from posture_optimization.config import load_config
from posture_optimization.movl import MovLValidator

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def backend():
    b = TesseractBackend(load_config(ROOT/"config/default.json"))
    yield b
    b.close()


def test_library_ik_fk_and_complete_movl(backend):
    start = backend.base_pose[:3, 3]+[0, -.4, .8]
    q = backend.ik(start, np.eye(3), np.zeros(6))
    assert q is not None
    assert np.linalg.norm(backend.fk(q)[:3, 3]-start) < 1e-4
    result = MovLValidator(backend, backend.config["validation"]).plan(q, start+[.02, 0, 0], np.eye(3))
    assert result.success, result.reason
    assert result.min_inverse_condition >= .06


def test_jacobian_matches_independent_fk_difference(backend):
    q = np.array([.4, .5, -.7, .3, .8, -.2])
    jacobian = backend.jacobian(q)
    epsilon = 1e-6
    for i in range(6):
        a, b = q.copy(), q.copy()
        a[i] -= epsilon
        b[i] += epsilon
        difference = (backend.fk(b)[:3, 3]-backend.fk(a)[:3, 3])/(2*epsilon)
        assert np.allclose(jacobian[:3, i], difference, atol=1e-5)


def test_descartes_cartesian_planner_and_independent_verification(backend):
    start = backend.base_pose[:3, 3]+[0, -.4, .8]
    q = backend.ik(start, np.eye(3), np.zeros(6))
    positions = start+np.linspace(0, 1, 5)[1:, None]*[.02, 0, 0]
    trajectory = backend.descartes(q, positions, np.eye(3))
    result = MovLValidator(backend, backend.config["validation"]).validate(trajectory, q, start+[.02, 0, 0], np.eye(3))
    assert result.success, result.reason


def test_collision_obstacle_blocks_tcp():
    c = load_config(ROOT/"config/default.json")
    c["obstacles"] = [{"kind": "box", "size_mm": [200, 200, 200], "pose_mm_deg": [0, -400, 750, 0, 0, 0]}]
    b = TesseractBackend(c)
    try:
        q = b.ik([0, -.4, .8], np.eye(3), np.zeros(6))
        assert q is not None
        assert b.collision(q)
    finally:
        b.close()
