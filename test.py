"""
Local comprehensive smoke test runner for the mathcore package.

What this script does:
1. Imports every Python module under mathcore (module-level coverage).
2. Runs targeted smoke checks across core, complex, statistics,
   probability, optimization, differential, and utils.
3. Prints a summary and exits non-zero if any check fails.

Run:
    python test.py
"""

import importlib
import traceback
from pathlib import Path


def discover_modules(package_root: Path):
    modules = []
    for py in package_root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        rel = py.relative_to(package_root.parent).with_suffix("")
        modules.append(".".join(rel.parts))
    return sorted(modules)


def run_module_import_tests(results):
    root = Path(__file__).resolve().parent
    package_root = root / "mathcore"
    modules = discover_modules(package_root)

    print(f"[IMPORT] Found {len(modules)} modules under mathcore")
    for mod in modules:
        try:
            importlib.import_module(mod)
            results.append((f"import:{mod}", True, ""))
        except Exception:
            results.append((f"import:{mod}", False, traceback.format_exc()))


def check(results, name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except Exception:
        results.append((name, False, traceback.format_exc()))


def run_smoke_tests(results):
    import mathcore

    # Core
    check(results, "core:factorial", lambda: mathcore.factorial(6) == 720)
    check(results, "core:gcd", lambda: mathcore.gcd(54, 24) == 6)
    check(results, "core:matrix", lambda: mathcore.Matrix([[1, 2], [3, 4]]).determinant())
    check(results, "core:number_theory", lambda: mathcore.is_prime_miller_rabin(101))
    check(results, "core:combinatorics", lambda: mathcore.catalan_number(5))

    # Geometry / Calculus / Algebra
    check(
        results,
        "core:geometry_distance",
        lambda: mathcore.distance(mathcore.Point(0, 0), mathcore.Point(3, 4)),
    )
    check(results, "core:quadratic", lambda: mathcore.solve_quadratic(1, -3, 2))
    check(results, "core:derivative", lambda: mathcore.derivative(lambda x: x * x, 2.0))

    # Complex
    check(results, "complex:numbers", lambda: mathcore.Complex(1, 2).magnitude())
    check(results, "complex:analysis", lambda: mathcore.complex_exp(mathcore.Complex(1, 1)))
    check(results, "complex:transforms", lambda: mathcore.fft([1.0, 0.0, -1.0, 0.0]))

    # Statistics
    check(results, "stats:descriptive", lambda: mathcore.DescriptiveStatistics.mean([1, 2, 3, 4]))
    check(results, "stats:inference", lambda: mathcore.ttest_one_sample([1, 2, 3, 4, 5], 3.0))
    check(results, "stats:regression", lambda: mathcore.OLSRegression().fit([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0]))
    check(results, "stats:time_series", lambda: mathcore.acf([1, 2, 3, 4, 5, 6], 2))

    # Probability
    check(results, "prob:distributions", lambda: mathcore.NormalDistribution(0, 1).pdf(0))
    check(results, "prob:markov", lambda: mathcore.MarkovChain([[0.9, 0.1], [0.2, 0.8]]).steady_state())
    check(results, "prob:mc", lambda: mathcore.simulate_pi(10000))

    # Optimization
    check(results, "opt:algorithms", lambda: mathcore.GradientDescent())
    check(
        results,
        "opt:linear_programming",
        lambda: mathcore.simplex([3, 2], [[1, 1], [1, 0], [0, 1]], [4, 2, 3]),
    )

    # Differential
    check(results, "diff:ode", lambda: mathcore.ODESolver())
    check(
        results,
        "diff:bvp",
        lambda: mathcore.finite_difference_bvp(
            lambda _x: 0.0,
            lambda _x: 0.0,
            lambda _x: -1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            n=10,
        ),
    )

    # Utils
    check(results, "utils:step_display", lambda: mathcore.StepTracker())


def print_summary(results):
    total = len(results)
    failed = [r for r in results if not r[1]]
    passed = total - len(failed)

    print("\n=== TEST SUMMARY ===")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failed)}")
    print(f"Total : {total}")

    if failed:
        print("\n=== FAILURES ===")
        for name, _, err in failed:
            print(f"\n[{name}]")
            print(err)

    return 1 if failed else 0


def main():
    results = []
    run_module_import_tests(results)
    run_smoke_tests(results)
    code = print_summary(results)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
