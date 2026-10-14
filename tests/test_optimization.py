from itertools import combinations
import numpy as np
from posture_optimization.optimize import coverage, optimize


def test_maximum_union_matches_exhaustive_solution():
    rng = np.random.default_rng(3)
    matrix = rng.random((8, 17)) < .25
    settings = {"k": 3, "method": "both", "de_maxiter": 20, "de_popsize": 5, "time_limit_s": 10}
    result = optimize(matrix, settings, 42)
    exhaustive = max(coverage(matrix, list(indices)).sum() for indices in combinations(range(8), 3))
    assert result["covered_count"] == exhaustive
    assert result["optimal"]
    assert len(set(result["indices"])) == 3
    assert result["de"]["evaluations"] > 0


def test_empty_candidates_and_k_larger_than_pool():
    settings = {"k": 3, "method": "differential_evolution", "de_maxiter": 5, "de_popsize": 5, "time_limit_s": 1}
    assert optimize(np.zeros((0, 5)), settings, 42)["status"] == "NO_CANDIDATES"
    result = optimize(np.zeros((2, 5)), settings, 42)
    assert len(result["indices"]) == 2
    assert result["coverage"] == 0
