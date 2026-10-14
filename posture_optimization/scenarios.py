"""Search arbitrary task scenarios and select shared joint postures."""
import copy
import json
import math
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .backend import TesseractBackend
from .config import load_config, validate
from .geometry import contains, sample_region
from .movl import MovLValidator
from .pipeline import candidates
from .optimize import optimize, coverage


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def target_weights(counts, scenario_weights):
    """Exact rational normalization: each scenario contributes its requested weight."""
    fractions = [Fraction(str(w))/int(n) for n, w in zip(counts, scenario_weights)]
    if not fractions or any(f <= 0 for f in fractions):
        raise ValueError('Scenario weights and target counts must be positive')
    denominator = math.lcm(*(f.denominator for f in fractions))
    integers = [f.numerator*(denominator//f.denominator) for f in fractions]
    divisor = math.gcd(*integers)
    integers = [i//divisor for i in integers]
    if sum(n*w for n, w in zip(counts, integers)) >= 2**60:
        raise ValueError('Scenario normalization exceeds integer precision; simplify weights/counts')
    return np.concatenate([np.full(n, w, dtype=np.int64) for n, w in zip(counts, integers)])


def search(manifest_path, output, log=print):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    base = load_config(manifest_path.parent/manifest['base_config'])
    entries = manifest['scenarios']
    if not entries or len({e['name'] for e in entries}) != len(entries):
        raise ValueError('Provide nonempty scenarios with unique names')
    configs = [merge(base, e.get('overrides', {})) for e in entries]
    for config in configs:
        config['robot']['urdf'] = str((manifest_path.parent/config['robot']['urdf']).resolve())
        validate(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    backends = []
    try:
        for config in configs:
            backends.append(TesseractBackend(config))
        first = backends[0]
        if any(b.joint_names != first.joint_names or not np.array_equal(b.limits, first.limits) for b in backends):
            raise ValueError('Shared postures require identical joint names/order and joint limits')
        if any(c['robot']['urdf'] != configs[0]['robot']['urdf'] for c in configs):
            raise ValueError('Scenarios must use the same robot URDF; vary base, TCP and task regions')
        rotations = [Rotation.from_euler('xyz', c['orientation_rpy_deg'], degrees=True).as_matrix() for c in configs]
        validators = [MovLValidator(b, c['validation']) for b, c in zip(backends, configs)]
        pool = []
        for entry, config, backend, validator, rotation in zip(entries, configs, backends, validators, rotations):
            log(f"Generating candidates: {entry['name']}")
            generated, _ = candidates(backend, validator, config, rotation, log)
            for q in generated:
                if not any(np.max(abs(q-existing)) < np.deg2rad(base['candidates']['dedup_deg']) for existing in pool):
                    pool.append(q)
        qs = np.asarray(pool).reshape(-1, len(first.joint_names))
        points = [sample_region(c['target'], c.get('reference_axis'), b.base_pose[:3, 3]) for c, b in zip(configs, backends)]
        if any(not len(p) for p in points):
            raise ValueError('A scenario has no sampled targets')
        matrices, eligibility = [], []
        for index, (entry, config, backend, validator, rotation, targets) in enumerate(zip(entries, configs, backends, validators, rotations, points)):
            matrix = np.zeros((len(qs), len(targets)), bool)
            eligible = np.zeros(len(qs), bool)
            for i, q in enumerate(qs):
                pose = backend.fk(q)
                angle = np.rad2deg(Rotation.from_matrix(rotation.T @ pose[:3, :3]).magnitude())
                eligible[i] = (validator.state_reason(q)[0] == 'ok' and
                    angle <= config['validation']['orientation_tolerance_deg'] and
                    contains(pose[:3, 3], config['initial'], config.get('reference_axis'), backend.base_pose[:3, 3])[0])
                if eligible[i]:
                    for j, target in enumerate(targets):
                        matrix[i, j] = validator.plan(q.copy(), target, rotation).success
            matrices.append(matrix)
            eligibility.append(eligible)
            log(f"Evaluated {entry['name']}: {int(eligible.sum())}/{len(qs)} eligible starts")
            directory = output/f'scenario_{index:03d}'
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/'coverage.npz', A=matrix, targets_m=targets, candidates_rad=qs, eligible=eligible)
            (directory/'resolved_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        weights = target_weights([len(p) for p in points], [e.get('weight', 1) for e in entries])
        combined = np.concatenate(matrices, axis=1)
        settings = base['optimization']
        shared = optimize(combined, {**settings, 'k': 1}, base['seed'], weights)
        library = optimize(combined, settings, base['seed'], weights)
        best_index = max(range(len(entries)), key=lambda s: float(matrices[s].sum(axis=1).max()/len(points[s])) if len(qs) else 0)
        scene_best = optimize(matrices[best_index], {**settings, 'k': 1}, base['seed'])
        best_qs = scene_best['indices']
        best = {'scenario': entries[best_index]['name'], 'scenario_index': best_index,
                'configuration': configs[best_index], 'selection': scene_best,
                'joints_deg': np.rad2deg(qs[best_qs]).tolist(),
                'tcp_pose_world_mm_deg': [np.r_[backends[best_index].fk(q)[:3, 3]*1000,
                    Rotation.from_matrix(backends[best_index].fk(q)[:3, :3]).as_euler('xyz', degrees=True)].tolist() for q in qs[best_qs]]}
        def describe(selection):
            return {**selection, 'joints_deg': np.rad2deg(qs[selection['indices']]).tolist(),
                    'per_scenario': [{'name': e['name'], 'coverage': float(coverage(m, selection['indices']).mean()),
                                      'eligible_selected_count': int(valid[selection['indices']].sum())}
                                     for e, m, valid in zip(entries, matrices, eligibility)]}
        report = {'schema_version': 2, 'candidate_count': len(qs), 'joint_names': first.joint_names,
                  'best_scenario': best, 'shared_posture': describe(shared), 'posture_library': describe(library),
                  'objective': 'Weighted mean of sampled target coverage fractions; ineligible starts contribute zero',
                  'scope': 'Optimality applies to this finite scenario list and generated candidate pool'}
        np.savez_compressed(output/'combined.npz', A=combined, candidates_rad=qs, weights=weights)
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        log(json.dumps(report, indent=2))
        return report
    finally:
        for backend in backends:
            backend.close()
