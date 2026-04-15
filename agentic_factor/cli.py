from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import dump_yaml, load_config
from .demo import generate_demo_panel
from .pipeline import AgenticFactorPipeline
from .report import render_run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-factor", description="Local runnable package for agentic factor investing reproduction")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_demo = sub.add_parser("make-demo-data", help="Generate a synthetic daily equity panel.")
    p_demo.add_argument("--out", required=True); p_demo.add_argument("--n-assets", type=int, default=120); p_demo.add_argument("--start", default="2018-01-01"); p_demo.add_argument("--end", default="2024-12-31"); p_demo.add_argument("--seed", type=int, default=42)
    p_run = sub.add_parser("run", help="Run the agentic factor pipeline.")
    p_run.add_argument("--config", required=True); p_run.add_argument("--data", required=True); p_run.add_argument("--out", required=True); p_run.add_argument("--report", action="store_true")
    p_report = sub.add_parser("report", help="Render plots and markdown from an existing run folder.")
    p_report.add_argument("--run-dir", required=True)
    p_init = sub.add_parser("init-config", help="Write a starter config file to a chosen path.")
    p_init.add_argument("--template", choices=["paper_default", "demo_small"], default="demo_small"); p_init.add_argument("--out", required=True)
    return parser


def _copy_template(name: str, out: str | Path) -> None:
    here = Path(__file__).resolve().parent.parent
    cfg = load_config(here / "configs" / f"{name}.yaml")
    dump_yaml(cfg, out)


def main() -> None:
    parser = build_parser(); args = parser.parse_args()
    if args.cmd == "make-demo-data":
        print(generate_demo_panel(args.out, args.n_assets, args.start, args.end, args.seed)); return
    if args.cmd == "run":
        config = load_config(args.config); summary = AgenticFactorPipeline(config).run(args.data, args.out)
        if args.report:
            summary["report"] = str(render_run_report(args.out))
        print(json.dumps(summary, indent=2, ensure_ascii=False)); return
    if args.cmd == "report":
        print(render_run_report(args.run_dir)); return
    if args.cmd == "init-config":
        _copy_template(args.template, args.out); print(args.out); return


if __name__ == "__main__":
    main()
