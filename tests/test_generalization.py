import copy
import json
import threading
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path
import numpy as np
import pytest
from posture_optimization.backend import TesseractBackend
from posture_optimization.config import load_config, validate
from posture_optimization.geometry import contains, sample_region
from posture_optimization.monitor import make_server
from posture_optimization.movl import MovLValidator
from posture_optimization.optimize import optimize, coverage
from posture_optimization.scenarios import merge, target_weights, search

ROOT = Path(__file__).resolve().parents[1]


def test_regions_without_a_reference_axis():
    sphere = {'kind': 'sphere', 'center_mm': [0, 0, 0], 'radius_mm': 100, 'spacing_mm': 100}
    assert len(sample_region(sphere)) == 7
    points = {'kind': 'points', 'points_mm': [[0, 0, 0], [100, 200, 300], [0, 0, 0]]}
    assert len(sample_region(points)) == 2
    assert contains([[.1, .2, .3], [1, 1, 1]], points).tolist() == [True, False]
    config = load_config(ROOT/'config/default.json')
    config['target'] = points
    validate(config)


def test_region_axis_is_independent_and_half_side_explicit():
    region = {'kind': 'half_cylinder', 'axis': {'start_mm': [0, 0, 0], 'end_mm': [0, 0, 200]},
              'outer_radius_mm': 100, 'spacing_mm': 100, 'side': 'negative', 'split_normal': [1, 0, 0]}
    assert contains([[-.05, 0, .1], [.05, 0, .1]], region).tolist() == [True, False]
    assert contains(sample_region(region), region).all()


@pytest.mark.parametrize('method', ['cp_sat', 'both', 'differential_evolution'])
def test_weighted_optimization_against_exhaustive_search(method):
    matrix = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], bool)
    weights = np.array([1, 1, 20, 8])
    settings = {'k': 2, 'method': method, 'time_limit_s': 5, 'de_maxiter': 30, 'de_popsize': 5}
    result = optimize(matrix, settings, 42, weights)
    brute = max(int(weights @ coverage(matrix, list(indices))) for indices in combinations(range(4), 2))
    assert result['weighted_score'] == brute == 28
    assert result['covered_count'] == 2


def test_scenario_normalization_does_not_favor_dense_grids():
    weights = target_weights([2, 10], [1, 1])
    assert weights[:2].sum() == weights[2:].sum()
    weights = target_weights([2, 10], [1, 2])
    assert weights[2:].sum() == 2*weights[:2].sum()
    original = {'robot': {'tcp': [0]*6, 'name': 'demo'}}
    result = merge(original, {'robot': {'tcp': [1]*6}})
    assert original['robot']['tcp'] == [0]*6
    assert result['robot'] == {'tcp': [1]*6, 'name': 'demo'}


def test_redundant_seven_joint_urdf_and_movl(tmp_path):
    config = load_config(ROOT/'config/default.json')
    tree = ET.parse(config['robot']['urdf'])
    root = tree.getroot()
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', str((ROOT/'models'/mesh.get('filename')).resolve()))
    children = {j.find('child').get('link') for j in root.findall('joint')}
    original_root = next(link.get('name') for link in root.findall('link') if link.get('name') not in children)
    ET.SubElement(root, 'link', name='redundant_mount')
    joint = ET.SubElement(root, 'joint', name='redundant_joint', type='revolute')
    ET.SubElement(joint, 'parent', link='redundant_mount')
    ET.SubElement(joint, 'child', link=original_root)
    ET.SubElement(joint, 'axis', xyz='0 0 1')
    ET.SubElement(joint, 'limit', lower='-3.14', upper='3.14', effort='1', velocity='1')
    path = tmp_path/'redundant.urdf'
    tree.write(path)
    config['robot']['urdf'] = str(path)
    backend = TesseractBackend(config)
    try:
        assert len(backend.joint_names) == 7
        target = np.array([0, -.4, .8])
        q = backend.ik(target, np.eye(3), np.zeros(7))
        assert q is not None and q.shape == (7,)
        assert backend.jacobian(q).shape == (6, 7)
        result = MovLValidator(backend, config['validation']).plan(q, target+[.01, 0, 0], np.eye(3))
        assert result.success, result.reason
        assert result.trajectory.shape[1] == 7
    finally:
        backend.close()


def test_monitor_http_metrics_and_bad_inputs():
    server = make_server(load_config(ROOT/'config/default.json'), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(url+'/') as response:
            assert b'Robot Singularity Monitor' in response.read()
        with urllib.request.urlopen(url+'/api/meta') as response:
            assert len(json.load(response)['joint_names']) == 6
        q = server.backend.ik([0, -.4, .8], np.eye(3), np.zeros(6))
        request = urllib.request.Request(url+'/api/metrics', data=json.dumps({'joints_deg': np.rad2deg(q).tolist()}).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
            assert data['state'] == 'ok'
            assert len(data['singular_values']) == 6
            assert np.allclose(data['tcp_world_mm'], [0, -400, 800], atol=.1)
        with urllib.request.urlopen(url+'/api/latest') as response:
            assert json.load(response) == data
        for body in ({'joints_deg': [0]}, {'joints_deg': [999]*6}, {'joints_deg': [float('nan')]*6}):
            request = urllib.request.Request(url+'/api/metrics', data=json.dumps(body).encode())
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            assert error.value.code == 400
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        server.backend.close()


def test_scenario_search_with_different_target_counts(tmp_path):
    config = load_config(ROOT/'config/default.json')
    config['candidates']['count'] = 2
    (tmp_path/'base.json').write_text(json.dumps(config))
    manifest = {'base_config': 'base.json', 'scenarios': [
        {'name': 'box', 'overrides': {}},
        {'name': 'points', 'overrides': {'target': {'kind': 'points', 'points_mm': [[0, -400, 800], [20, -400, 800]]}}}]}
    (tmp_path/'scenarios.json').write_text(json.dumps(manifest))
    result = search(tmp_path/'scenarios.json', tmp_path/'result', log=lambda *a, **k: None)
    selected = result['shared_posture']
    assert selected['optimal']
    assert selected['weighted_coverage'] == pytest.approx(np.mean([s['coverage'] for s in selected['per_scenario']]))
    assert (tmp_path/'result/scenario_001/coverage.npz').is_file()
