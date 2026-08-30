"""
soo/cli.py — argparse entry point.

    uv run soo run --dry-run          # full grid with a stub client, no API calls
    uv run soo manipcheck             # can the cues be recovered at all?
    uv run soo run --limit 2          # two real scenarios, all providers
    uv run soo run                    # the pilot
    uv run soo judge
    uv run soo crossjudge
    uv run soo analyse

Handler imports are deliberately local so that a missing optional dependency
(torch, for the white-box path) cannot break unrelated commands.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .analyse import run_analysis
from .config import MANIPCHECK_FLOOR, MAX_CONCURRENCY, N_SAMPLES
from .figures import make_figures
from .judge import manipulation_check, score
from .register import run_register
from .robustness import main as run_robustness
from .run_behavioural import run
from .validate import export_for_labelling, extend_for_labelling, score_agreement
from .whitebox import run_whitebox


def _cmd_run(args: argparse.Namespace) -> int:
    asyncio.run(
        run(
            dry_run=args.dry_run,
            limit=args.limit,
            only_models=args.models,
            samples=args.samples,
            max_concurrency=args.concurrency,
            control=args.control,
            dose=args.dose,
        )
    )
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    asyncio.run(score(dry_run=args.dry_run, cross=False, max_concurrency=args.concurrency))
    return 0


def _cmd_crossjudge(args: argparse.Namespace) -> int:
    asyncio.run(score(dry_run=args.dry_run, cross=True, max_concurrency=args.concurrency))
    return 0


def _cmd_manipcheck(args: argparse.Namespace) -> int:
    result = asyncio.run(manipulation_check(dry_run=args.dry_run, limit=args.limit))
    if not args.dry_run and result["accuracy"] < MANIPCHECK_FLOOR:
        print(
            f"[manipcheck] FAIL: accuracy {result['accuracy']:.3f} < floor {MANIPCHECK_FLOOR}. "
            "The cues are not recoverable, so a null result would carry no information. "
            "Fix the cues before spending on the full run.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_validate_export(args: argparse.Namespace) -> int:
    export_for_labelling(n=args.n)
    return 0


def _cmd_validate_extend(args) -> int:
    extend_for_labelling(args.n)
    return 0


def _cmd_validate_score(args: argparse.Namespace) -> int:
    score_agreement()
    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    asyncio.run(run_register(dry_run=args.dry_run, n_scenarios=args.n))
    return 0


def _cmd_whitebox(args: argparse.Namespace) -> int:
    run_whitebox(limit=args.limit, skip_generation=args.skip_generation)
    return 0


def _cmd_robustness(args) -> int:
    return run_robustness()


def _cmd_analyse(args: argparse.Namespace) -> int:
    run_analysis()
    return 0


def _cmd_figures(args: argparse.Namespace) -> int:
    make_figures()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soo", description="Self-Over-Other: vulnerable-user asymmetry experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser, *, dry_run: bool = True, concurrency: bool = True) -> None:
        if dry_run:
            sub.add_argument("--dry-run", action="store_true", help="use a stub client; costs nothing")
        if concurrency:
            sub.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)

    run_parser = subparsers.add_parser("run", help="generate the response grid")
    add_common(run_parser)
    run_parser.add_argument("--limit", type=int, default=None, help="use only the first N scenarios")
    run_parser.add_argument("--models", nargs="*", default=None, help="restrict to these model keys")
    run_parser.add_argument("--samples", type=int, default=None)
    run_parser.add_argument("--control", action="store_true",
                            help="run only the structural positive-control arm")
    run_parser.add_argument("--dose", action="store_true",
                            help="run the reduced stated-vs-structural grid over EXTRA_MODELS")
    run_parser.set_defaults(func=_cmd_run)

    judge_parser = subparsers.add_parser("judge", help="score responses against the rubric")
    add_common(judge_parser)
    judge_parser.set_defaults(func=_cmd_judge)

    cross_parser = subparsers.add_parser("crossjudge", help="re-score a subsample with a different family")
    add_common(cross_parser)
    cross_parser.set_defaults(func=_cmd_crossjudge)

    manip_parser = subparsers.add_parser("manipcheck", help="recover the vulnerability cue from the prompt alone")
    add_common(manip_parser, concurrency=False)
    manip_parser.add_argument("--limit", type=int, default=None)
    manip_parser.set_defaults(func=_cmd_manipcheck)

    export_parser = subparsers.add_parser("validate-export", help="write a CSV for hand-labelling")
    export_parser.add_argument("--n", type=int, default=None)
    export_parser.set_defaults(func=_cmd_validate_export)

    extend_parser = subparsers.add_parser("validate-extend", help="append a second labelling batch")
    extend_parser.add_argument("--n", type=int, default=150)
    extend_parser.set_defaults(func=_cmd_validate_extend)

    agree_parser = subparsers.add_parser("validate-score", help="judge-vs-human kappa per rubric item")
    agree_parser.set_defaults(func=_cmd_validate_score)

    register_parser = subparsers.add_parser("register", help="formal-vs-casual register robustness check")
    add_common(register_parser, concurrency=False)
    register_parser.add_argument("--n", type=int, default=10, help="scenarios to include")
    register_parser.set_defaults(func=_cmd_register)

    whitebox_parser = subparsers.add_parser("whitebox", help="self-vs-user activation distance on a local model")
    whitebox_parser.add_argument("--limit", type=int, default=None)
    whitebox_parser.add_argument("--skip-generation", action="store_true", help="probe only; reuse existing behaviour")
    whitebox_parser.set_defaults(func=_cmd_whitebox)

    analyse_parser = subparsers.add_parser("analyse", help="regenerate every number in the write-up")
    analyse_parser.set_defaults(func=_cmd_analyse)

    robust_parser = subparsers.add_parser("robustness", help="mixed effects, paired, leave-one-domain, composition, judges")
    robust_parser.set_defaults(func=_cmd_robustness)

    figures_parser = subparsers.add_parser("figures", help="regenerate the figures")
    figures_parser.set_defaults(func=_cmd_figures)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # `--samples` defaults to the configured N_SAMPLES when not given.
    if getattr(args, "samples", None) is None and args.command == "run":
        args.samples = N_SAMPLES

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
