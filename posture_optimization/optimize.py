"""Maximum coverage: SciPy differential evolution and optional CP-SAT proof."""
import numpy as np
from scipy.optimize import differential_evolution


def coverage(matrix, indices):
    return np.any(matrix[indices], axis=0) if len(indices) else np.zeros(matrix.shape[1], bool)


def greedy(matrix, k, weights=None):
    weights = np.ones(matrix.shape[1], dtype=np.int64) if weights is None else weights
    selected = []
    for _ in range(k):
        available = [i for i in range(len(matrix)) if i not in selected]
        selected.append(max(available, key=lambda i: int(weights @ coverage(matrix, selected+[i]))))
    return selected


def optimize(matrix, settings, seed, weights=None):
    matrix = np.asarray(matrix, bool)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Coverage matrix needs at least one target point")
    m, n = matrix.shape
    weights = np.ones(n, dtype=np.int64) if weights is None else np.asarray(weights)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0) or np.any(weights != np.floor(weights)):
        raise ValueError("Weights require positive integers, one per target")
    if sum(int(x) for x in weights) >= 2**60:
        raise ValueError("Integer objective exceeds supported precision")
    weights = weights.astype(np.int64)
    total_weight = int(weights.sum())
    def score(indices):
        return int(weights @ coverage(matrix, indices))
    if m == 0:
        return {"indices": [], "covered_count": 0, "coverage": 0., "status": "NO_CANDIDATES", "optimal": False, "weighted_score": 0, "weighted_coverage": 0.}
    k = min(settings["k"], m)
    baseline = greedy(matrix, k, weights)
    details = {}
    selected = baseline
    if settings["method"] in ("both", "differential_evolution"):
        # Integer indices, duplicate penalty; keep greedy incumbent if DE is worse.
        def objective(x):
            indices = np.unique(np.rint(x).astype(int))
            return -score(indices)+(k-len(indices))*(total_weight+1)
        if k < m:
            de = differential_evolution(objective, [(0, m-1)]*k, integrality=np.ones(k, bool),
                                        seed=seed, maxiter=settings["de_maxiter"], popsize=settings["de_popsize"],
                                        polish=False, x0=baseline, tol=0, atol=0)
            candidate = list(dict.fromkeys(np.rint(de.x).astype(int).tolist()))
            candidate += [i for i in baseline if i not in candidate]
            candidate += [i for i in range(m) if i not in candidate]
            candidate = candidate[:k]
            if score(candidate) >= score(selected):
                selected = candidate
            details["de"] = {"evaluations": int(de.nfev), "converged": bool(de.success),
                             "covered_count": int(coverage(matrix, selected).sum())}
        else:
            details["de"] = {"evaluations": 0, "converged": True, "covered_count": int(coverage(matrix, selected).sum())}
    status, optimal = "HEURISTIC", k == m
    if settings["method"] in ("both", "cp_sat"):
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        x = [model.new_bool_var(f"candidate_{i}") for i in range(m)]
        y = [model.new_bool_var(f"point_{j}") for j in range(n)]
        model.add(sum(x) == k)
        for j in range(n):
            covering = np.flatnonzero(matrix[:, j])
            if len(covering):
                model.add_max_equality(y[j], [x[int(i)] for i in covering])
            else:
                model.add(y[j] == 0)
        for i in range(m):
            model.add_hint(x[i], int(i in selected))
        model.maximize(sum(int(weights[j])*y[j] for j in range(n)))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = settings["time_limit_s"]
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed
        code = solver.solve(model)
        status = solver.status_name(code)
        if code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            candidate = [i for i in range(m) if solver.value(x[i])]
            if score(candidate) >= score(selected):
                selected = candidate
            optimal = code == cp_model.OPTIMAL
        details["cp_sat"] = {"status": status, "upper_bound": float(solver.best_objective_bound)}
    count = int(coverage(matrix, selected).sum())
    return {"indices": selected, "covered_count": count, "coverage": count/n,
            "weighted_score": score(selected), "weighted_coverage": score(selected)/total_weight,
            "status": status, "optimal": optimal, "requested_k": settings["k"], "actual_k": k, **details}
