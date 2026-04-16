from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import iohgnbg
import numpy as np
from ioh import logger

GNBG_INSTANCES_FOLDER = "benchmarks/gnbg/official"
GNBG_BASE_BUDGET = 2000

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "problem_ids": [1, 2],
        "reps": 1,
        "budget_scale": 0.02,
        "parallel_workers": 1,
    },
    "search": {
        "problem_ids": list(range(1, 10)),
        "reps": 1,
        "budget_scale": 0.1,
        "parallel_workers": 6,
    },
    "timing": {
        "problem_ids": list(range(1, 25)),
        "reps": 3,
        "budget_scale": 1.0,
        "parallel_workers": 1,
    },
    "final": {
        "problem_ids": list(range(1, 25)),
        "reps": 31,
        "budget_scale": 1.0,
        "parallel_workers": 8,
    },
}


class OverBudgetException(RuntimeError):
    pass


class _BoundsView:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub


@dataclass
class CaseResult:
    ok: bool
    fid: int
    rep: int
    score: float | None = None
    auc: float | None = None
    elapsed_s: float | None = None
    budget: int | None = None
    dim: int | None = None
    fes: int | None = None
    error: str | None = None


class AOCLogger:
    """
    Minimal AOC logger replacement.
    Stores the incumbent best value after each evaluation.
    """

    def __init__(self):
        self.values: list[float] = []
        self.best_values: list[float] = []

    def __call__(self, _, log_info):
        value = float(log_info.raw_y_best)
        self.values.append(value)
        if self.best_values:
            self.best_values.append(min(self.best_values[-1], value))
        else:
            self.best_values.append(value)

    def reset(self, *_):
        self.values.clear()
        self.best_values.clear()


# NOTE:
# This is a lightweight fallback approximation of the AOCC/AOC-style score used
# in your larger LLaMEA pipeline. It produces a normalized score in [0, 1] where
# higher is better, but it is not guaranteed to match your original misc.py
# helpers exactly. For the competition repo, this is often good enough for a
# quick-search loop; for official numbers, swap this back to your exact misc.py.
def correct_aoc(problem, log: AOCLogger, budget: int, upper: float = 1e2) -> float:
    if not log.best_values:
        return 0.0

    raw = np.asarray(log.best_values, dtype=float)
    raw = np.minimum(raw, upper)

    optimum = getattr(getattr(problem, "optimum", None), "y", None)
    if optimum is None:
        optimum = np.min(raw)

    normalized = (upper - raw) / max(1e-12, upper - float(optimum))
    normalized = np.clip(normalized, 0.0, 1.0)

    if normalized.size < budget:
        pad_value = normalized[-1]
        normalized = np.pad(normalized, (0, budget - normalized.size), constant_values=pad_value)

    return float(np.mean(normalized[:budget]))


class IOHProblemAdapter:
    def __init__(self, problem, budget: int):
        self._problem = problem
        self.dim = _problem_dim(problem)
        self.budget = int(budget)
        self.lower, self.upper = _problem_bounds(problem, self.dim)
        self.bounds = _BoundsView(self.lower, self.upper)
        self.evaluations = 0
        self.best_values: list[float] = []

    def __call__(self, x):
        if self.evaluations >= self.budget:
            raise OverBudgetException(
                f"Evaluation budget exceeded: {self.evaluations} >= {self.budget}"
            )

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != self.dim:
            raise ValueError(
                f"Expected a {self.dim}-dimensional point, got shape {x.shape}"
            )

        y = self._problem(x)
        y_arr = np.asarray(y)
        if y_arr.size != 1:
            raise ValueError(f"Expected scalar objective value, got shape {y_arr.shape}")

        self.evaluations += 1
        value = float(y_arr.item())
        if self.best_values:
            self.best_values.append(min(self.best_values[-1], value))
        else:
            self.best_values.append(value)
        return value

    def reset(self):
        self.evaluations = 0
        self.best_values.clear()
        return self._problem.reset()

    def __getattr__(self, name: str):
        return getattr(self._problem, name)


def _normalize_problems(problems) -> list[Any]:
    if isinstance(problems, list):
        return problems
    if isinstance(problems, tuple):
        return list(problems)
    try:
        return list(problems)
    except TypeError:
        return [problems]


def _load_ioh_problem(fid: int):
    last_error: Exception | None = None

    for problem_indices, selector in (
        ([fid], lambda items: items[0]),
        ((fid,), lambda items: items[0]),
        (24, lambda items: items[fid - 1]),
    ):
        try:
            problems = _normalize_problems(
                iohgnbg.get_problems(
                    problem_indices=problem_indices,
                    instances_folder=GNBG_INSTANCES_FOLDER,
                )
            )
            if problems:
                return selector(problems)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc

    detail = f" Last error: {type(last_error).__name__}: {last_error}" if last_error else ""
    raise RuntimeError(
        f"Could not load IOH GNBG problem f{fid} from {GNBG_INSTANCES_FOLDER}.{detail}"
    )


def _problem_dim(problem) -> int:
    meta_data = getattr(problem, "meta_data", None)
    if meta_data is not None and hasattr(meta_data, "n_variables"):
        return int(meta_data.n_variables)
    if hasattr(problem, "dim"):
        return int(problem.dim)
    raise AttributeError("Could not determine problem dimensionality.")


def _problem_bounds(problem, dim: int) -> tuple[np.ndarray, np.ndarray]:
    bounds = getattr(problem, "bounds", None)
    if bounds is not None and hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        return np.asarray(bounds.lb, dtype=float), np.asarray(bounds.ub, dtype=float)

    if hasattr(problem, "lower") and hasattr(problem, "upper"):
        return np.asarray(problem.lower, dtype=float), np.asarray(problem.upper, dtype=float)

    lower = np.full(dim, -5.0, dtype=float)
    upper = np.full(dim, 5.0, dtype=float)
    return lower, upper


def _run_single_case(module_name: str, class_name: str, fid: int, rep: int, budget_scale: float):
    np.random.seed(rep)

    try:
        module = __import__(module_name, fromlist=[class_name])
        algorithm_cls = getattr(module, class_name)
    except Exception as exc:
        return CaseResult(
            ok=False,
            fid=fid,
            rep=rep,
            error=f"Could not import {module_name}.{class_name}: {type(exc).__name__}: {exc}",
        )

    try:
        problem = _load_ioh_problem(fid)
        scaled_budget = max(1, int(GNBG_BASE_BUDGET * budget_scale))
        wrapped_problem = IOHProblemAdapter(problem, scaled_budget)

        algorithm = algorithm_cls(budget=scaled_budget, dim=wrapped_problem.dim)

        import time
        t0 = time.perf_counter()
        try:
            algorithm(wrapped_problem)
        except OverBudgetException:
            pass
        elapsed_s = time.perf_counter() - t0

        log = AOCLogger()
        log.best_values = list(wrapped_problem.best_values)
        auc = float(correct_aoc(problem, log, scaled_budget))
        log.reset(problem)
        problem.reset()

        return CaseResult(
            ok=True,
            fid=fid,
            rep=rep,
            score=auc,
            auc=auc,
            elapsed_s=elapsed_s,
            budget=scaled_budget,
            dim=wrapped_problem.dim,
            fes=wrapped_problem.evaluations,
        )
    except Exception as exc:
        return CaseResult(
            ok=False,
            fid=fid,
            rep=rep,
            error=f"Runtime error on IOH GNBG f{fid}, rep {rep}: {type(exc).__name__}: {exc}",
        )


def evaluate_candidate(
    module_name: str = "candidate",
    class_name: str = "Algorithm",
    profile: str = "quick",
    workers: int | None = None,
    budget_scale: float | None = None,
) -> dict[str, Any]:
    preset = PROFILE_PRESETS[profile].copy()
    if workers is not None:
        preset["parallel_workers"] = max(1, int(workers))
    if budget_scale is not None:
        preset["budget_scale"] = float(budget_scale)

    cases = [(fid, rep) for fid in preset["problem_ids"] for rep in range(preset["reps"])]

    if preset["parallel_workers"] <= 1 or len(cases) <= 1:
        results = [
            _run_single_case(module_name, class_name, fid, rep, preset["budget_scale"])
            for fid, rep in cases
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=preset["parallel_workers"]) as executor:
            future_map = {
                executor.submit(_run_single_case, module_name, class_name, fid, rep, preset["budget_scale"]): (fid, rep)
                for fid, rep in cases
            }
            for future in as_completed(future_map):
                results.append(future.result())
        results.sort(key=lambda item: (item.fid, item.rep))

    failures = [r for r in results if not r.ok]
    run_scores = [r.score for r in results if r.ok and r.score is not None]

    summary = {
        "profile": profile,
        "problem_ids": preset["problem_ids"],
        "reps": preset["reps"],
        "budget_scale": preset["budget_scale"],
        "parallel_workers": preset["parallel_workers"],
        "base_budget": GNBG_BASE_BUDGET,
        "instances_folder": GNBG_INSTANCES_FOLDER,
        "score_mean": float(np.mean(run_scores)) if run_scores else None,
        "score_std": float(np.std(run_scores)) if run_scores else None,
        "total_fes": int(sum(r.fes or 0 for r in results)),
        "failures": len(failures),
        "results": [asdict(r) for r in results],
    }

    if failures:
        summary["first_error"] = failures[0].error

    return summary


def write_json(path: str | Path, payload: dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
