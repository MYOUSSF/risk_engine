#!/usr/bin/env python3
# risk_engine/run_pipeline.py
"""
Market Risk Engine — Pipeline Runner
=====================================
Executes modules 1-5 in dependency order with a single command.

Usage:
    python run_pipeline.py              # run all five modules
    python run_pipeline.py --from 3     # start from Module 3 (VaR Engine)

Module order:
    1  Data Ingestion   — fetch prices, compute log returns
    2  Portfolio        — mark-to-market positions, daily P&L
    3  VaR Engine       — Historical / Parametric / Monte Carlo VaR
    4  Backtesting      — Kupiec, Christoffersen, Basel traffic light
    5  Stress Testing   — historical replays, hypothetical shocks, reverse stress
"""

import argparse
import importlib
import os
import sys
import time
import traceback

# Add the project's parent directory so `import risk_engine` resolves correctly.
# This mirrors the sys.path pattern used in every module file.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ── Module registry ───────────────────────────────────────────────────────────

MODULES = [
    (1, "Data Ingestion", "risk_engine.modules.data_ingestion"),
    (2, "Portfolio",       "risk_engine.modules.portfolio"),
    (3, "VaR Engine",      "risk_engine.modules.var_engine"),
    (4, "Backtesting",     "risk_engine.modules.backtesting"),
    (5, "Stress Testing",  "risk_engine.modules.stress_testing"),
]

# ── Terminal formatting ───────────────────────────────────────────────────────

W = 62   # output column width


def _line(char="═"):
    return char * W


def _centre(text, char=" "):
    pad = max(0, W - len(text))
    return char * (pad // 2) + text + char * (pad - pad // 2)


def _print_header(from_module: int) -> None:
    print(_line())
    print(_centre("Market Risk Engine — Pipeline"))
    print(_line())
    print()
    if from_module > 1:
        skipped = ", ".join(name for n, name, _ in MODULES if n < from_module)
        print(f"  --from {from_module}: skipping modules 1–{from_module - 1}")
        print(f"  ({skipped})")
        print()


def _print_module_header(num: int, name: str) -> None:
    label = f"  Running Module {num}: {name}"
    print(_line("─"))
    print(label)
    print(_line("─"))


def _print_module_result(num: int, name: str, elapsed: float, error: str | None) -> None:
    if error is None:
        mark, note = "✓", f"{elapsed:.1f}s"
    else:
        mark, note = "✗", "FAILED"
    print(f"\n  {mark}  Module {num}: {name}  —  {note}")
    print()


def _print_summary(results: list, total_elapsed: float) -> None:
    n_run       = sum(1 for *_, err, skipped in results if not skipped)
    n_succeeded = sum(1 for *_, err, skipped in results if not skipped and err is None)
    n_failed    = n_run - n_succeeded
    first_fail  = next((n for n, *_, err, skipped in results
                        if not skipped and err is not None), None)

    print(_line())
    print("  Pipeline Summary")
    print(_line("─"))

    for num, name, elapsed, error, skipped in results:
        if skipped:
            mark = "—"
            tag  = f"skipped"
        elif error is None:
            mark = "✓"
            tag  = f"{elapsed:.1f}s"
        else:
            mark = "✗"
            tag  = "FAILED"
        print(f"  {mark}  Module {num}  {name:<20}  {tag}")

    print(_line("─"))
    if n_failed == 0:
        print(f"  All {n_succeeded} modules succeeded  ({total_elapsed:.1f}s total)")
    else:
        print(f"  {n_succeeded} succeeded, {n_failed} failed  ({total_elapsed:.1f}s total)")
        if first_fail is not None:
            print(f"  Tip: fix errors above, then re-run with --from {first_fail}")

    print()
    print("  Launch the dashboard:")
    print("    streamlit run risk_engine/modules/dashboard.py")
    print(_line())


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(from_module: int) -> int:
    """Run modules from `from_module` onward. Returns exit code (0 = all OK)."""
    _print_header(from_module)

    results       = []
    pipeline_start = time.perf_counter()

    for num, name, module_path in MODULES:
        if num < from_module:
            results.append((num, name, 0.0, None, True))   # skipped
            continue

        _print_module_header(num, name)
        t0    = time.perf_counter()
        error = None

        try:
            mod  = importlib.import_module(module_path)
            conn = mod.run()
            # Close the connection so the next module opens a clean handle.
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            # Print the full traceback so the user can diagnose the root cause.
            print()
            traceback.print_exc()
            print()

        elapsed = time.perf_counter() - t0
        results.append((num, name, elapsed, error, False))
        _print_module_result(num, name, elapsed, error)

    total_elapsed = time.perf_counter() - pipeline_start
    _print_summary(results, total_elapsed)

    any_failed = any(err for _, _, _, err, skipped in results if not skipped)
    return 1 if any_failed else 0


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run_pipeline.py",
        description="Run the Market Risk Engine pipeline (modules 1-5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python run_pipeline.py           # full pipeline\n"
            "  python run_pipeline.py --from 3  # restart from VaR Engine\n"
            "  python run_pipeline.py --from 4  # restart from Backtesting\n"
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_module",
        type=int,
        default=1,
        metavar="N",
        help="start from module N (1-5, default: 1)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if not 1 <= args.from_module <= 5:
        parser.error(f"--from must be between 1 and 5, got {args.from_module}")

    sys.exit(run_pipeline(args.from_module))


if __name__ == "__main__":
    main()
