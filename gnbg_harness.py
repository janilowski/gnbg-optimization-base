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
GNBG_BASE_BUDGET = 20000

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "problem_ids": [1, 2],
        "reps": 1,
        "budget_scale": 0.02,
        "parallel_workers": 1,
    },
    "search": {
        "problem_ids": list(range(1, 25)),
        "reps": 3,
        "budget_scale": 0.2,
        "parallel_workers": 8,
    },
    "hard": {
        "problem_ids": [9, 13, 3, 10, 14],
        "reps": 5,
        "budget_scale": 10.0,
        "parallel_workers": 8,
    },
    "timing": {
        "problem_ids": list(range(1, 25)),
        "reps": 3,
        "budget_scale": 10.0,
        "parallel_workers": 8,
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
    seed: int | None = None
    score: float | None = None
    auc: float | None = None
    anchor_random_score: float | None = None
    anchor_local_score: float | None = None
    delta_vs_random: float | None = None
    delta_vs_local: float | None = None
    elapsed_s: float | None = None
    budget: int | None = None
    dim: int | None = None
    fes: int | None = None
    best_value: float | None = None
    optimum: float | None = None
    gap_to_optimum: float | None = None
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


def correct_aoc(
    problem,
    log: AOCLogger,
    budget: int,
    lower: float = 1e-8,
    upper: float = 1e8,
) -> float:
    if not log.best_values:
        return 0.0

    optimum = getattr(getattr(problem, "optimum", None), "y", None)
    if optimum is None:
        optimum = float(np.min(log.best_values))

    raw = np.asarray(log.best_values, dtype=float)
    gap = np.clip(raw - float(optimum), lower, upper)

    log_u, log_l = np.log10(upper), np.log10(lower)
    normalized = (log_u - np.log10(gap)) / (log_u - log_l)
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


def _derive_seed(seed_base: int, fid: int, rep: int) -> int:
    return int(seed_base + fid * 10007 + rep)


def _run_random_baseline(problem, budget: int, seed: int) -> list[float]:
    dim = _problem_dim(problem)
    lower, upper = _problem_bounds(problem, dim)
    rng = np.random.default_rng(seed)
    best_values: list[float] = []
    best_y = float("inf")
    for _ in range(int(budget)):
        x = rng.uniform(lower, upper, size=dim)
        y = float(np.asarray(problem(x)).item())
        best_y = min(best_y, y)
        best_values.append(best_y)
    return best_values


def _run_local_baseline(problem, budget: int, seed: int) -> list[float]:
    dim = _problem_dim(problem)
    lower, upper = _problem_bounds(problem, dim)
    span = np.maximum(1e-12, upper - lower)
    sigma = 0.15 * span
    rng = np.random.default_rng(seed + 1)

    x = rng.uniform(lower, upper, size=dim)
    best_y = float(np.asarray(problem(x)).item())
    best_values: list[float] = [best_y]

    remaining = int(budget) - 1
    while remaining > 0:
        z = x + rng.normal(0.0, sigma, size=dim)
        z = np.clip(z, lower, upper)
        y = float(np.asarray(problem(z)).item())
        if y < best_y:
            x = z
            best_y = y
            sigma = np.minimum(0.5 * span, sigma * 1.03)
        else:
            sigma = np.maximum(1e-12, sigma * 0.99)
        best_values.append(best_y)
        remaining -= 1
    return best_values


def _run_single_case(
    module_name: str,
    class_name: str,
    fid: int,
    rep: int,
    budget_scale: float,
    seed_base: int,
    with_anchors: bool,
):
    seed = _derive_seed(seed_base, fid, rep)
    np.random.seed(seed)

    try:
        module = __import__(module_name, fromlist=[class_name])
        algorithm_cls = getattr(module, class_name)
    except Exception as exc:
        return CaseResult(
            ok=False,
            fid=fid,
            rep=rep,
            seed=seed,
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

        best_value = float(wrapped_problem.best_values[-1]) if wrapped_problem.best_values else None
        optimum_obj = getattr(problem, "optimum", None)
        optimum = float(optimum_obj.y) if optimum_obj is not None and hasattr(optimum_obj, "y") else None
        gap_to_optimum = (
            float(best_value - optimum)
            if best_value is not None and optimum is not None
            else None
        )

        log.reset(problem)
        problem.reset()

        anchor_random_score = None
        anchor_local_score = None
        if with_anchors:
            random_trace = _run_random_baseline(problem, scaled_budget, seed)
            random_log = AOCLogger()
            random_log.best_values = random_trace
            anchor_random_score = float(correct_aoc(problem, random_log, scaled_budget))
            random_log.reset(problem)
            problem.reset()

            local_trace = _run_local_baseline(problem, scaled_budget, seed)
            local_log = AOCLogger()
            local_log.best_values = local_trace
            anchor_local_score = float(correct_aoc(problem, local_log, scaled_budget))
            local_log.reset(problem)
            problem.reset()

        return CaseResult(
            ok=True,
            fid=fid,
            rep=rep,
            seed=seed,
            score=auc,
            auc=auc,
            anchor_random_score=anchor_random_score,
            anchor_local_score=anchor_local_score,
            delta_vs_random=(auc - anchor_random_score) if anchor_random_score is not None else None,
            delta_vs_local=(auc - anchor_local_score) if anchor_local_score is not None else None,
            elapsed_s=elapsed_s,
            budget=scaled_budget,
            dim=wrapped_problem.dim,
            fes=wrapped_problem.evaluations,
            best_value=best_value,
            optimum=optimum,
            gap_to_optimum=gap_to_optimum,
        )
    except Exception as exc:
        return CaseResult(
            ok=False,
            fid=fid,
            rep=rep,
            seed=seed,
            error=f"Runtime error on IOH GNBG f{fid}, rep {rep}: {type(exc).__name__}: {exc}",
        )


def evaluate_candidate(
    module_name: str = "candidate",
    class_name: str = "Algorithm",
    profile: str = "quick",
    workers: int | None = None,
    budget_scale: float | None = None,
    reps: int | None = None,
    seed_base: int = 12345,
    with_anchors: bool = True,
) -> dict[str, Any]:
    preset = PROFILE_PRESETS[profile].copy()
    if workers is not None:
        preset["parallel_workers"] = max(1, int(workers))
    if budget_scale is not None:
        preset["budget_scale"] = float(budget_scale)
    if reps is not None:
        preset["reps"] = max(1, int(reps))

    cases = [(fid, rep) for fid in preset["problem_ids"] for rep in range(preset["reps"])]

    if preset["parallel_workers"] <= 1 or len(cases) <= 1:
        results = [
            _run_single_case(
                module_name,
                class_name,
                fid,
                rep,
                preset["budget_scale"],
                seed_base,
                with_anchors,
            )
            for fid, rep in cases
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=preset["parallel_workers"]) as executor:
            future_map = {
                executor.submit(
                    _run_single_case,
                    module_name,
                    class_name,
                    fid,
                    rep,
                    preset["budget_scale"],
                    seed_base,
                    with_anchors,
                ): (fid, rep)
                for fid, rep in cases
            }
            for future in as_completed(future_map):
                results.append(future.result())
        results.sort(key=lambda item: (item.fid, item.rep))

    failures = [r for r in results if not r.ok]
    run_scores = [r.score for r in results if r.ok and r.score is not None]
    delta_vs_random = [r.delta_vs_random for r in results if r.ok and r.delta_vs_random is not None]
    delta_vs_local = [r.delta_vs_local for r in results if r.ok and r.delta_vs_local is not None]
    gaps = [r.gap_to_optimum for r in results if r.ok and r.gap_to_optimum is not None]

    trimmed_mean = None
    if run_scores:
        sorted_scores = np.sort(np.asarray(run_scores, dtype=float))
        trim = int(0.1 * sorted_scores.size)
        core = sorted_scores[trim:-trim] if (trim > 0 and sorted_scores.size > 2 * trim) else sorted_scores
        trimmed_mean = float(np.mean(core))

    per_fid: dict[int, list[float]] = {}
    per_fid_gap: dict[int, list[float]] = {}
    for item in results:
        if not item.ok:
            continue
        if item.score is not None:
            per_fid.setdefault(item.fid, []).append(float(item.score))
        if item.gap_to_optimum is not None:
            per_fid_gap.setdefault(item.fid, []).append(float(item.gap_to_optimum))

    per_problem = {
        str(fid): {
            "score_mean": float(np.mean(vals)),
            "score_std": float(np.std(vals)),
            "gap_mean": float(np.mean(per_fid_gap[fid])) if fid in per_fid_gap else None,
            "gap_std": float(np.std(per_fid_gap[fid])) if fid in per_fid_gap else None,
            "n": len(vals),
        }
        for fid, vals in sorted(per_fid.items())
    }

    summary = {
        "profile": profile,
        "problem_ids": preset["problem_ids"],
        "reps": preset["reps"],
        "budget_scale": preset["budget_scale"],
        "parallel_workers": preset["parallel_workers"],
        "seed_base": int(seed_base),
        "with_anchors": bool(with_anchors),
        "base_budget": GNBG_BASE_BUDGET,
        "instances_folder": GNBG_INSTANCES_FOLDER,
        "score_mean": float(np.mean(run_scores)) if run_scores else None,
        "score_std": float(np.std(run_scores)) if run_scores else None,
        "score_median": float(np.median(run_scores)) if run_scores else None,
        "score_trimmed_mean": trimmed_mean,
        "delta_vs_random_mean": float(np.mean(delta_vs_random)) if delta_vs_random else None,
        "delta_vs_local_mean": float(np.mean(delta_vs_local)) if delta_vs_local else None,
        "gap_mean": float(np.mean(gaps)) if gaps else None,
        "gap_std": float(np.std(gaps)) if gaps else None,
        "total_fes": int(sum(r.fes or 0 for r in results)),
        "failures": len(failures),
        "per_problem": per_problem,
        "results": [asdict(r) for r in results],
    }

    if failures:
        summary["first_error"] = failures[0].error

    return summary


def write_json(path: str | Path, payload: dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
