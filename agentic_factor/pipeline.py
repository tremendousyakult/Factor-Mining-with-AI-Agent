from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .agent import HeuristicAgent, paper_seed_library, traditional_baseline_library
from .aggregation import CompositeResult, evaluate_composite, linear_composite, lgbm_composite, make_feature_frame, quarterly_cost_table, risk_adjusted_alpha
from .config import dump_yaml
from .data import apply_screens, compute_base_panel, load_panel, split_is_oos
from .evaluation import FactorEvaluation, evaluate_library
from .recipe import FactorRecipe
from .utils import make_output_dir, write_jsonl


class AgenticFactorPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def run(self, data_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        out_dir = make_output_dir(output_dir)
        dump_yaml(self.config, out_dir / "resolved_config.yaml")
        raw = load_panel(data_path, self.config)
        screened = apply_screens(raw, self.config)
        panel = compute_base_panel(screened, self.config)
        panel.to_csv(out_dir / "panel_prepared.csv")
        panel_is, panel_oos = split_is_oos(panel, self.config)
        discovery_mode = self.config["agent"].get("mode", "agentic")
        if discovery_mode == "paper_seed":
            promoted = paper_seed_library(); evals_is = evaluate_library(promoted, panel_is, self.config)
        elif discovery_mode == "traditional_baseline":
            promoted = traditional_baseline_library(); evals_is = evaluate_library(promoted, panel_is, self.config)
        else:
            promoted, evals_is, logs = self._run_agentic_loop(panel_is); write_jsonl(logs, out_dir / "round_logs.jsonl")
        self._write_factor_evaluations(evals_is, out_dir / "factor_metrics_is.csv")
        promoted = [ev.recipe for ev in evals_is if ev.decision == "promote"] if discovery_mode == "agentic" else promoted
        if not promoted:
            raise RuntimeError("No factors were promoted. Lower the gate thresholds or use the paper_seed mode.")
        promoted_is = evaluate_library(promoted, panel_is, self.config)
        promoted_oos = evaluate_library(promoted, panel_oos, self.config)
        self._write_factor_evaluations(promoted_is, out_dir / "promoted_factor_metrics_is.csv")
        self._write_factor_evaluations(promoted_oos, out_dir / "promoted_factor_metrics_oos.csv")
        write_jsonl([rec.to_dict() for rec in promoted], out_dir / "promoted_library.jsonl")
        feature_is = make_feature_frame(promoted_is, panel_is); feature_oos = make_feature_frame(promoted_oos, panel_oos)
        feature_is.to_csv(out_dir / "factor_scores_is.csv"); feature_oos.to_csv(out_dir / "factor_scores_oos.csv")
        factor_names = [ev.recipe.factor_id for ev in promoted_is]
        linear_score, linear_artifacts = linear_composite(feature_oos, factor_names)
        linear_result = evaluate_composite("linear", linear_score, panel_oos, float(self.config["costs"].get("one_way_bps", 3.0)))
        lgbm_score, lgbm_artifacts = lgbm_composite(feature_is, feature_oos, factor_names, self.config)
        lgbm_result = evaluate_composite("lgbm", lgbm_score, panel_oos, float(self.config["costs"].get("one_way_bps", 3.0)))
        self._write_composite(linear_result, out_dir / "linear"); self._write_composite(lgbm_result, out_dir / "lgbm")
        self._write_model_meta(linear_artifacts, out_dir / "linear_model.json"); self._write_model_meta(lgbm_artifacts, out_dir / "lgbm_model.json")
        alpha_df = self._maybe_benchmarks(linear_result, out_dir)
        summary = {"discovery_mode": discovery_mode, "n_promoted": len(promoted), "promoted_factor_ids": [r.factor_id for r in promoted], "linear_summary": linear_result.summary, "lgbm_summary": lgbm_result.summary, "benchmark_alpha_rows": int(alpha_df.shape[0]) if alpha_df is not None else 0}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    def _run_agentic_loop(self, panel_is: pd.DataFrame):
        agent = HeuristicAgent(seed=int(self.config["agent"].get("seed", 42)))
        rounds = int(self.config["agent"].get("rounds", 5)); per_round = int(self.config["agent"].get("candidates_per_round", 20))
        promoted: dict[str, FactorRecipe] = {}; last_evals = []; logs = []
        for round_idx in range(1, rounds + 1):
            candidates = agent.generate(round_idx, per_round); evals = evaluate_library(candidates, panel_is, self.config)
            round_metrics = []; promoted_this_round = []
            for ev in evals:
                row = {"round": round_idx, "factor_id": ev.recipe.factor_id, "name": ev.recipe.name, "expression": ev.recipe.expression, "rationale": ev.recipe.rationale, "family": ev.recipe.family, "decision": ev.decision, **ev.metrics}
                round_metrics.append(row); logs.append(row)
                if ev.decision == "promote":
                    promoted[ev.recipe.factor_id] = ev.recipe; promoted_this_round.append(ev.recipe)
            agent.absorb_round(round_metrics, promoted_this_round); last_evals = evals
        final_recipes = list(promoted.values())
        final_evals = evaluate_library(final_recipes, panel_is, self.config) if final_recipes else last_evals
        return final_recipes, final_evals, logs

    @staticmethod
    def _write_factor_evaluations(evals: list[FactorEvaluation], path: Path) -> None:
        rows = [{"factor_id": ev.recipe.factor_id, "name": ev.recipe.name, "family": ev.recipe.family, "expression": ev.recipe.expression, "rationale": ev.recipe.rationale, "decision": ev.decision, **ev.metrics} for ev in evals]
        pd.DataFrame(rows).to_csv(path, index=False)

    @staticmethod
    def _write_model_meta(artifacts, path: Path) -> None:
        meta = {"feature_names": artifacts.feature_names}
        if artifacts.model is not None:
            meta["feature_importances"] = getattr(artifacts.model, "feature_importances_", []).tolist()
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _write_composite(result: CompositeResult, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        result.decile_returns.to_csv(out_dir / "decile_returns.csv")
        pd.DataFrame({"gross_ls": result.ls_returns, "net_ls": result.net_ls_returns, "turnover": result.turnover}).to_csv(out_dir / "spread_returns.csv")
        quarterly_cost_table(result).to_csv(out_dir / "quarterly_cost_table.csv", index=False)

    def _maybe_benchmarks(self, linear_result: CompositeResult, out_dir: Path) -> pd.DataFrame | None:
        bcfg = self.config.get("benchmarks") or {}
        file_path = bcfg.get("file")
        if not file_path:
            return None
        df = pd.read_csv(file_path); date_col = bcfg.get("date_col", "date"); df[date_col] = pd.to_datetime(df[date_col]); df = df.set_index(date_col).sort_index()
        groups = bcfg.get("groups", {"CAPM": ["MKT"], "FF3": ["MKT", "SMB", "HML"], "FF5": ["MKT", "SMB", "HML", "RMW", "CMA"], "FF6": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]})
        alpha_df = risk_adjusted_alpha(linear_result.ls_returns, df, groups, risk_free_col=bcfg.get("risk_free_col"))
        if not alpha_df.empty:
            alpha_df.to_csv(out_dir / "linear_benchmark_alpha.csv", index=False)
        return alpha_df
