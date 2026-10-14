"""Independent MovL validation, including joint-interpolated subsegments."""
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ValidationResult:
    success: bool
    reason: str
    trajectory: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    max_position_error_mm: float = 0.
    max_orientation_error_deg: float = 0.
    min_inverse_condition: float = 1.

    def report(self):
        return {"success": self.success, "reason": self.reason, "samples": len(self.trajectory),
                "max_position_error_mm": self.max_position_error_mm,
                "max_orientation_error_deg": self.max_orientation_error_deg,
                "min_inverse_condition": self.min_inverse_condition}


class MovLValidator:
    def __init__(self, backend, settings):
        self.backend = backend
        self.settings = settings

    def state_reason(self, q):
        cache = getattr(self, '_state_cache', None)
        if cache is None:
            return self._state_reason(q)
        key = (np.shape(q), np.asarray(q, dtype=float).tobytes())
        if key not in cache:
            cache[key] = self._state_reason(q)
        return cache[key]

    def _state_reason(self, q):
        if np.shape(q) != (len(self.backend.limits),) or not np.isfinite(q).all():
            return "invalid_joints", 0.
        if np.any(q < self.backend.limits[:, 0]-1e-9) or np.any(q > self.backend.limits[:, 1]+1e-9):
            return "joint_limits", 0.
        j = self.backend.jacobian(q).copy()
        j[:3] /= self.settings["characteristic_length_m"]
        singular = np.linalg.svd(j, compute_uv=False)
        ratio = float(singular[-1]/singular[0]) if len(singular) == 6 and singular[0] > 0 else 0.
        if ratio < self.settings["inverse_condition_min"]:
            return "singularity", ratio
        if self.settings.get("collision_check", False) and self.backend.collision(q):
            return "collision", ratio
        return "ok", ratio

    def plan(self, initial, target, rotation):
        # Memoize only within one path: the independent validation still checks
        # every interpolation point, while identical FK/Jacobian inputs reuse
        # their exact results. No rounded keys or cross-path state are used.
        self._state_cache, self._fk_cache = {}, {}
        try:
            return self._plan(initial, target, rotation)
        finally:
            self._state_cache = self._fk_cache = None

    def _fk(self, q):
        cache = getattr(self, '_fk_cache', None)
        if cache is None:
            return self.backend.fk(q)
        key = np.asarray(q, dtype=float).tobytes()
        if key not in cache:
            cache[key] = self.backend.fk(q)
        return cache[key]

    def _plan(self, initial, target, rotation):
        q0 = np.asarray(initial, float)
        target = np.asarray(target, float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Target requires three finite metre values")
        reason, ratio = self.state_reason(q0)
        if reason != "ok":
            return ValidationResult(False, reason, min_inverse_condition=ratio)
        start = self._fk(q0)[:3, 3]
        count = max(1, int(np.ceil(np.linalg.norm(target-start)/(self.settings["cartesian_step_mm"]/1000))))
        positions = start + np.linspace(0, 1, count+1)[1:, None]*(target-start)
        if self.settings.get("planner", "seeded_ik") == "descartes":
            try:
                trajectory = self.backend.descartes(q0, positions, rotation)
            except RuntimeError:
                return ValidationResult(False, "planner_failed")
        else:
            trajectory = [q0]
            for position in positions:
                q = self.backend.ik(position, rotation, trajectory[-1])
                if q is None:
                    return ValidationResult(False, "ik_failed", np.asarray(trajectory))
                # Reject immediately before additional inverse solves.
                if np.max(abs(q-trajectory[-1])) > np.deg2rad(self.settings["max_joint_jump_deg"]):
                    return ValidationResult(False, "joint_jump", np.asarray(trajectory))
                reason, ratio = self.state_reason(q)
                if reason != "ok":
                    return ValidationResult(False, reason, np.asarray(trajectory), min_inverse_condition=ratio)
                trajectory.append(q)
        return self.validate(trajectory, q0, target, rotation)

    def validate(self, trajectory, initial, target, rotation):
        qs = np.asarray(trajectory, float)
        result = ValidationResult(False, "invalid_trajectory", qs)
        if qs.ndim != 2 or qs.shape[1] != len(self.backend.limits) or len(qs) < 2 or not np.isfinite(qs).all():
            return result
        if np.max(abs(qs[0]-initial)) > 1e-6:
            result.reason = "wrong_start"
            return result
        if np.max(abs(np.diff(qs, axis=0))) > np.deg2rad(self.settings["max_joint_jump_deg"]):
            result.reason = "joint_jump"
            return result
        start = self._fk(np.asarray(initial, float))[:3, 3]
        delta = np.asarray(target)-start
        length2 = float(delta @ delta)
        previous_progress = -1e-8
        checked = []
        for a, b in zip(qs[:-1], qs[1:]):
            parts = max(1, int(np.ceil(np.max(abs(b-a))/np.deg2rad(self.settings["joint_step_deg"]))))
            # Each endpoint is checked, even for zero-length segments.
            for alpha in np.linspace(0, 1, parts+1):
                q = a+alpha*(b-a)
                reason, ratio = self.state_reason(q)
                result.min_inverse_condition = min(result.min_inverse_condition, ratio)
                if reason != "ok":
                    result.reason = reason
                    return result
                pose = self._fk(q)
                progress = float((pose[:3, 3]-start) @ delta/length2) if length2 > 1e-16 else 0.
                closest = start+np.clip(progress, 0, 1)*delta
                error = np.linalg.norm(pose[:3, 3]-closest)*1000
                angle = np.rad2deg(Rotation.from_matrix(rotation.T @ pose[:3, :3]).magnitude())
                result.max_position_error_mm = max(result.max_position_error_mm, float(error))
                result.max_orientation_error_deg = max(result.max_orientation_error_deg, float(angle))
                if error > self.settings["position_tolerance_mm"]:
                    result.reason = "line_error"
                    return result
                if angle > self.settings["orientation_tolerance_deg"]:
                    result.reason = "orientation_error"
                    return result
                if progress < previous_progress-1e-5:
                    result.reason = "non_monotonic"
                    return result
                previous_progress = progress
                if self.settings.get("collision_check", False) and checked and self.backend.segment_collision(checked[-1], q):
                    result.reason = "segment_collision"
                    return result
                checked.append(q)
        endpoint_error = np.linalg.norm(self._fk(qs[-1])[:3, 3]-target)*1000
        if endpoint_error > self.settings["position_tolerance_mm"]:
            result.reason = "endpoint_error"
            return result
        result.success, result.reason = True, "ok"
        result.trajectory = np.asarray(checked)
        return result
