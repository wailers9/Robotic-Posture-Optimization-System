import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .config import load_config

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT/"config/default.json").is_file():
    ROOT = Path(sys.prefix)/"share/robotic-posture-optimizer"


def main():
    parser = argparse.ArgumentParser(description="Configurable robot posture and Cartesian coverage optimizer")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "doctor", "validate", "search-scenarios", "monitor"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path, default=ROOT/("config/scenarios.json" if command == "search-scenarios" else "config/default.json"))
        p.add_argument("--output", type=Path, default=Path.cwd()/"outputs"/command)
        if command == "run":
            p.add_argument("--show", action="store_true")
        if command == "validate":
            p.add_argument("--joints-deg", nargs="+", type=float, required=True)
            p.add_argument("--target-mm", nargs=3, type=float, required=True)
        if command == "monitor":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    try:
        if args.command == "search-scenarios":
            from .scenarios import search
            search(args.config, args.output)
            return
        config = load_config(args.config)
        if args.command == "monitor":
            from .monitor import serve
            serve(config, args.host, args.port)
            return
        if args.command == "run":
            from .pipeline import run
            run(config, args.output, args.show)
            return
        from .backend import TesseractBackend
        from .movl import MovLValidator
        backend = TesseractBackend(config)
        try:
            rotation = Rotation.from_euler("xyz", config["orientation_rpy_deg"], degrees=True).as_matrix()
            validator = MovLValidator(backend, config["validation"])
            args.output.mkdir(parents=True, exist_ok=True)
            if args.command == "validate":
                joints = np.deg2rad(args.joints_deg)
                result = validator.plan(joints, np.asarray(args.target_mm)/1000, rotation)
                report = result.report()
                np.savez_compressed(args.output/"trajectory.npz", joints_rad=result.trajectory)
                (args.output/"validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(report, indent=2))
                if not result.success:
                    sys.exit(2)
                return
            # Use the configured task's initial region, rather than robot-specific probes.
            from .pipeline import candidates
            pool, rejected = candidates(backend, validator, config, rotation)
            chosen = pool[0] if len(pool) else None
            if chosen is None:
                raise RuntimeError("Doctor could not find a valid diagnostic start")
            target = backend.fk(chosen)[:3, 3]+[.02, 0, 0]
            result = validator.plan(chosen, target, rotation)
            descartes_error = None
            descartes_report = None
            try:
                positions = backend.fk(chosen)[:3, 3]+np.linspace(0, 1, 5)[1:, None]*[.02, 0, 0]
                trajectory = backend.descartes(chosen, positions, rotation)
                descartes_report = validator.validate(trajectory, chosen, target, rotation).report()
            except (RuntimeError, TypeError, AttributeError) as error:
                descartes_error = str(error)
            report = {"backend": "tesseract", "joint_names": backend.joint_names,
                      "initial_joints_deg": np.rad2deg(chosen).tolist(), "target_mm": (target*1000).tolist(),
                      "seeded_ik_movl": result.report(), "descartes": descartes_report, "descartes_error": descartes_error}
            (args.output/"doctor.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            if not result.success:
                sys.exit(2)
        finally:
            backend.close()
    except (ValueError, RuntimeError, ImportError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
