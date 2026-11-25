"""Command line: `reelint check video.mp4`, `reelint rules`."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, banner
from .analysis import analyze
from .config import PROFILES, load_config
from .media import MediaError
from .report import Style, to_json, to_text
from .rules import ERROR, RULES, WARNING, lint

EXIT_OK, EXIT_FINDINGS, EXIT_FAILURE = 0, 1, 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reelint", description="A linter for short-form video.")
    parser.add_argument("--version", action="store_true", help="Show the version and exit")
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="Lint one or more videos")
    check.add_argument("files", nargs="+", help="Video files to check")
    check.add_argument("--profile", choices=sorted(PROFILES), default="reels",
                       help="Platform limits to check against (default: reels)")
    check.add_argument("--format", choices=["text", "json"], default="text", dest="output")
    check.add_argument("--config", help="Path to reelint.toml (default: ./reelint.toml if present)")
    check.add_argument("--disable", action="append", default=[], metavar="RULE",
                       help="Skip a rule or a whole category (hook, pacing, audio, ...). Repeatable; commas allowed")
    check.add_argument("--fail-on", choices=["error", "warning", "never"], default="error",
                       help="Exit with status 1 when findings at this level exist (default: error)")

    hook = sub.add_parser("hook", help="Profile the opening seconds (text, changes, sound, faces)")
    hook.add_argument("files", nargs="+", help="Video files to profile")
    hook.add_argument("--window", type=float, default=3.0, help="Seconds to treat as the hook (default: 3)")
    hook.add_argument("--format", choices=["text", "json"], default="text", dest="output")
    hook.add_argument("--csv", metavar="PATH",
                      help="Add results to a CSV log; fill in skip_rate yourself, it's kept on re-runs")
    hook.add_argument("--no-vision", action="store_true", help="Skip text and face detection (faster)")

    sub.add_parser("rules", help="List every rule")
    return parser


def _hook(args: argparse.Namespace) -> int:
    from . import hook, vision

    if args.window <= 0:
        print("reelint: --window must be positive", file=sys.stderr)
        return EXIT_FAILURE
    if not args.no_vision and not vision.available():
        print(f"reelint: text and face checks need {' and '.join(vision.missing())}: "
              "pip install reelint[vision]  (continuing without them)", file=sys.stderr)
    style = Style.for_stream(sys.stdout)
    profiles, status = [], EXIT_OK
    for path in args.files:
        try:
            profiles.append(hook.profile(path, window=args.window, use_vision=not args.no_vision))
        except MediaError as exc:
            print(f"reelint: {exc}", file=sys.stderr)
            status = EXIT_FAILURE
    if args.output == "json":
        data = [p.to_dict() for p in profiles]
        print(json.dumps(data[0] if len(args.files) == 1 and data else data, indent=2))
    else:
        print("\n\n".join(hook.to_text(p, style) for p in profiles))
    sys.stdout.flush()
    if args.csv and profiles:
        try:
            hook.write_csv(args.csv, profiles)
        except PermissionError:
            print(f"reelint: could not write {args.csv}; is it open in Excel?", file=sys.stderr)
            return EXIT_FAILURE
        print(f"\nLogged {len(profiles)} video(s) to {args.csv}. Add each one's skip rate in the skip_rate column.",
              file=sys.stderr)
    return status


def _list_rules() -> int:
    width = max(len(r) for r in RULES)
    for rule_id, entry in RULES.items():
        print(f"{rule_id.ljust(width)}  {entry.summary}")
    return EXIT_OK


def _check(args: argparse.Namespace) -> int:
    disable = [name.strip() for item in args.disable for name in item.split(",") if name.strip()]
    known = set(RULES) | {r.category for r in RULES.values()}
    unknown = [name for name in disable if name not in known]
    if unknown:
        print(f"reelint: unknown rule or category: {', '.join(unknown)} (see `reelint rules`)", file=sys.stderr)
        return EXIT_FAILURE
    try:
        config = load_config(args.profile, args.config, disable)
    except (OSError, ValueError) as exc:
        print(f"reelint: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    failing = {"error": {ERROR}, "warning": {ERROR, WARNING}, "never": set()}[args.fail_on]
    style = Style.for_stream(sys.stdout)
    reports, status = [], EXIT_OK
    for index, path in enumerate(args.files):
        try:
            analysis = analyze(path, config)
        except MediaError as exc:
            print(f"reelint: {exc}", file=sys.stderr)
            status = EXIT_FAILURE
            continue
        findings = lint(analysis, config)
        if any(f.severity in failing for f in findings) and status == EXIT_OK:
            status = EXIT_FINDINGS
        if args.output == "json":
            reports.append(to_json(analysis, findings))
        else:
            if index:
                print()
            print(to_text(analysis, findings, style))
    if args.output == "json":
        print(json.dumps(reports[0] if len(args.files) == 1 and reports else reports, indent=2))
    return status


def _show_wordmark() -> None:
    """Print the wordmark, but only when a person is watching: piped output stays plain."""
    if getattr(sys.stdout, "isatty", lambda: False)():
        print(banner.render(Style.for_stream(sys.stdout)), end="\n\n")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = _parser()
    args = parser.parse_args(argv)
    if args.version:
        _show_wordmark()
        print(f"reelint {__version__}")
        return EXIT_OK
    if args.command is None:
        _show_wordmark()
        parser.print_help()
        return EXIT_OK
    if args.command == "rules":
        return _list_rules()
    if args.command == "hook":
        return _hook(args)
    return _check(args)


if __name__ == "__main__":
    sys.exit(main())
