# Verification record

Verified on 2026-10-14 using Windows and Python 3.11, without robot hardware. This record describes the included demonstration and regression suite; it is not an acceptance certificate for an externally supplied robot model.

| Check | Result |
| --- | --- |
| `python -m pytest -q` | 26 tests passed |
| Single-task default configuration | 6 accepted starts; 45 target samples; selected union 40/45 (88.89%); CP-SAT `OPTIMAL` |
| Three-scenario example | 12 pooled starts; best scenario 45/45; one shared posture 130/135 (96.30%); CP-SAT `OPTIMAL` |
| Unequal target counts | Weighted objective matched the arithmetic mean of per-scenario coverage fractions for equal scenario weights |
| Weighted CP-SAT and DE | Small reference case matched exhaustive enumeration |
| Seven-joint bounded revolute chain | 6 × 7 Jacobian, IK/FK and verified short Cartesian path passed |
| HTTP monitor | Page, joint metadata, metrics, latest sample and invalid-input responses passed |
| Wheel build | Successful wheel containing configuration, mesh assets, model license and monitor HTML |
| Installed-wheel diagnostic outside checkout | Loaded packaged configuration/model and passed seeded-IK Cartesian validation |

The default `doctor` seeded-IK diagnostic produced a maximum line deviation of about 0.0182 mm and a minimum normalized inverse condition number of about 0.1008. The optional Descartes attempt was rejected for `joint_jump`; this result is retained in the report and documented in the README. A separate integration test verifies a Descartes path from another valid branch.

`OPTIMAL` refers to finite coverage matrices, not the continuous joint or workspace search. The examples use disabled collision checking. The suite independently checks configured obstacle contacts in a separate backend integration test.

Transient outputs, build directories and virtual environments are excluded from Git. Reproduce the results with the commands in the main README; numerical values can vary with runtime/library changes.
