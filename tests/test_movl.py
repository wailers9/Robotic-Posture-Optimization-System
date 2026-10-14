import numpy as np
from posture_optimization.movl import MovLValidator

SETTINGS = {"cartesian_step_mm": 10, "joint_step_deg": .5, "max_joint_jump_deg": 8,
            "collision_check": True,
            "position_tolerance_mm": .5, "orientation_tolerance_deg": .25,
            "characteristic_length_m": .5, "inverse_condition_min": .06}


class AnalyticFixture:
    """Synthetic chain to inject failures between valid endpoint states."""
    limits = np.tile([-2., 2.], (6, 1))

    def fk(self, q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        return pose

    def jacobian(self, q):
        return np.eye(6)

    def collision(self, q):
        return False

    def segment_collision(self, a, b):
        return False

    def ik(self, position, rotation, seed):
        return np.r_[position, [0, 0, 0]]


def test_restarts_from_same_start():
    backend = AnalyticFixture()
    validator = MovLValidator(backend, SETTINGS)
    initial = np.zeros(6)
    for target in ([.05, 0, 0], [0, .05, 0]):
        result = validator.plan(initial, target, np.eye(3))
        assert result.success
        assert np.allclose(result.trajectory[0], initial)
        assert np.allclose(result.trajectory[-1][:3], target)
    assert np.allclose(initial, 0)


def test_rejects_singularity_between_safe_endpoints():
    backend = AnalyticFixture()
    def jacobian(q):
        j = np.eye(6)
        j[-1, -1] = 0 if .04 < q[0] < .06 else 1
        return j
    backend.jacobian = jacobian
    validator = MovLValidator(backend, SETTINGS)
    qs = np.zeros((2, 6))
    qs[1, 0] = .1
    assert validator.validate(qs, qs[0], [.1, 0, 0], np.eye(3)).reason == "singularity"


def test_rejects_curved_joint_interpolation_between_cartesian_endpoints():
    backend = AnalyticFixture()
    original = backend.fk
    def fk(q):
        pose = original(q)
        pose[1, 3] = .01*np.sin(q[0]*np.pi/.1)
        return pose
    backend.fk = fk
    qs = np.zeros((2, 6))
    qs[1, 0] = .1
    result = MovLValidator(backend, SETTINGS).validate(qs, qs[0], [.1, 0, 0], np.eye(3))
    assert result.reason == "line_error"


def test_rejects_jump_limits_and_segment_collision():
    backend = AnalyticFixture()
    validator = MovLValidator(backend, SETTINGS)
    qs = np.zeros((2, 6))
    qs[1, 0] = .2
    assert validator.validate(qs, qs[0], [.2, 0, 0], np.eye(3)).reason == "joint_jump"
    assert validator.state_reason(np.ones(6)*3)[0] == "joint_limits"
    qs[1, 0] = .1
    backend.segment_collision = lambda a, b: True
    assert validator.validate(qs, qs[0], [.1, 0, 0], np.eye(3)).reason == "segment_collision"


def test_disabled_collision_constraint_skips_all_contact_queries():
    backend = AnalyticFixture()
    def unexpected(*args):
        raise AssertionError("Collision interface should not be called")
    backend.collision = backend.segment_collision = unexpected
    validator = MovLValidator(backend, {**SETTINGS, "collision_check": False})
    assert validator.plan(np.zeros(6), [.05, 0, 0], np.eye(3)).success
