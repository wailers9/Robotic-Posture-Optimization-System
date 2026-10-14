"""Robot-independent HTTP singularity monitor. It never commands a robot."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .backend import TesseractBackend
from .movl import MovLValidator


def metrics(backend, joints_deg):
    q = np.deg2rad(np.asarray(joints_deg, float))
    if q.shape != (len(backend.joint_names),) or not np.isfinite(q).all():
        raise ValueError('Provide one finite degree value per configured joint')
    if np.any(q < backend.limits[:, 0]) or np.any(q > backend.limits[:, 1]):
        raise ValueError('Joint values exceed URDF limits')
    settings = backend.config['validation']
    jacobian = backend.jacobian(q).copy()
    jacobian[:3] /= settings['characteristic_length_m']
    singular = np.linalg.svd(jacobian, compute_uv=False)
    ratio = float(singular[-1]/singular[0]) if singular[0] > 0 else 0.
    pose = backend.fk(q)
    return {'joint_names': backend.joint_names, 'joints_deg': np.rad2deg(q).tolist(),
            'singular_values': singular.tolist(), 'inverse_condition': ratio,
            'threshold': settings['inverse_condition_min'],
            'state': MovLValidator(backend, settings).state_reason(q)[0],
            'tcp_world_mm': (pose[:3, 3]*1000).tolist(),
            'tcp_world_rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist()}


def make_server(config, host='127.0.0.1', port=8790):
    backend = TesseractBackend(config)
    lock = threading.Lock()
    latest = None

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value):
            payload = json.dumps(value, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/':
                page = Path(__file__).with_name('monitor.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif self.path == '/api/meta':
                self.reply(200, {'joint_names': backend.joint_names,
                    'limits_deg': np.rad2deg(backend.limits).tolist(),
                    'characteristic_length_m': config['validation']['characteristic_length_m'],
                    'threshold': config['validation']['inverse_condition_min']})
            elif self.path == '/api/latest':
                with lock:
                    self.reply(200, latest)
            else:
                self.reply(404, {'error': 'Unknown endpoint'})

        def do_POST(self):
            nonlocal latest
            if self.path != '/api/metrics':
                self.reply(404, {'error': 'Unknown endpoint'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    raise ValueError('Request body must be 1..65536 bytes')
                body = json.loads(self.rfile.read(size))
                with lock:
                    latest = metrics(backend, body['joints_deg'])
                    self.reply(200, latest)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                self.reply(400, {'error': str(error)})

        def log_message(self, *args):
            pass

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except Exception:
        backend.close()
        raise
    server.backend = backend
    return server


def serve(config, host='127.0.0.1', port=8790):
    server = make_server(config, host, port)
    print(f'Singularity monitor: http://{host}:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.backend.close()
