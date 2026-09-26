"""Generate the authoritative paper-numbers table from the result JSONs.

Why this exists
---------------
Every number in the manuscript must be traceable to a file on disk. Hand-
transcribing them produced real errors: a headline P95 that was actually the
coord_only ablation row (18.58 instead of 18.65), and an LLM model name and
event count that match no run in this repository. This script reads the
artifacts and prints each citable number next to the file and JSON path it came
from, so the manuscript can be checked mechanically instead of from memory.

Run:    python -m multi_household.experiments.paper_numbers
Writes: reports/multi_household/paper_numbers.md  and  paper_numbers.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "multi_household"


def _load(name: str):
    p = REPORTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:                       # truncated / mid-write file
        print(f"  ! could not parse {name}: {exc}")
        return None


class Table:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, section, label, value, unit, src, path) -> None:
        if value is None:
            return
        self.rows.append({"section": section, "label": label, "value": value,
                          "unit": unit, "source": src, "path": path})


def collect() -> list[dict]:
    t = Table()

    # ---------- provenance ---------------------------------------------------
    man = _load("run_manifest.json")
    if man:
        f = "run_manifest.json"
        # the manifest is flat: run_utc / git_commit / houses / seed / ...
        t.add("provenance", "run timestamp (UTC)", man.get("run_utc"), "", f, "run_utc")
        t.add("provenance", "git commit", man.get("git_commit"), "", f, "git_commit")
        houses = man.get("houses")
        if houses:
            t.add("provenance", "households", len(houses), "count", f, "houses")
            t.add("provenance", "household IDs", ", ".join(map(str, houses)), "", f, "houses")
        for lab, key, unit in [("random seed", "seed", ""),
                               ("EV seed", "ev_seed", ""),
                               ("acceptance rate", "user_accept", ""),
                               ("forecast mode", "forecast_mode", ""),
                               ("EV coordinator enabled", "ev_smart", ""),
                               ("grid threshold G", "grid_threshold_w", "W"),
                               ("test-window steps", "test_steps", "10-min"),
                               ("test-window length", "test_days", "days"),
                               ("train/test split", "split_at", "")]:
            t.add("provenance", lab, man.get(key), unit, f, key)
        win = man.get("clean_window")
        if win:
            t.add("provenance", "clean window", " to ".join(win), "", f, "clean_window")
        for k, v in (man.get("versions") or {}).items():
            t.add("provenance", f"{k} version", v, "", f, f"versions/{k}")

    # ---------- headline ------------------------------------------------------
    ms = _load("metrics_summary.json")
    if ms:
        f = "metrics_summary.json"
        base_pk = (ms.get("baseline", {}).get("grid") or {}).get("agg_served_peak_kw")
        for arm in ("baseline", "independent", "coordinated"):
            node = ms.get(arm) or {}
            grid = node.get("grid") or {}
            t.add("headline", f"{arm} peak", grid.get("agg_served_peak_kw"), "kW", f,
                  f"{arm}/grid/agg_served_peak_kw")
            t.add("headline", f"{arm} P95", grid.get("agg_served_p95_kw"), "kW", f,
                  f"{arm}/grid/agg_served_p95_kw")
            t.add("headline", f"{arm} P95 reduction", grid.get("p95_reduction_pct"), "%", f,
                  f"{arm}/grid/p95_reduction_pct")
            pk = grid.get("agg_served_peak_kw")
            if base_pk and pk is not None:
                t.add("headline", f"{arm} peak reduction",
                      round(100 * (base_pk - pk) / base_pk, 2), "%", f,
                      f"derived from {arm}/grid/agg_served_peak_kw vs baseline")
            ec = node.get("energy_conservation") or {}
            t.add("headline", f"{arm} energy drift", ec.get("diff_pct"), "%", f,
                  f"{arm}/energy_conservation/diff_pct")
            t.add("headline", f"{arm} terminal buffer debt",
                  ec.get("terminal_buffer_debt_kwh"), "kWh", f,
                  f"{arm}/energy_conservation/terminal_buffer_debt_kwh")
            cm = node.get("comfort") or {}
            for lab, key in [("appliance decisions", "appliance_decisions"),
                             ("appliance accepted", "appliance_accepted"),
                             ("appliance accept rate", "appliance_accept_rate"),
                             ("EV decisions", "ev_decisions"),
                             ("EV accepted", "ev_accepted"),
                             ("EV accept rate", "ev_accept_rate"),
                             ("total decisions", "total_decisions"),
                             ("total accepted", "total_accepted"),
                             ("surface messages", "surface_messages_total")]:
                t.add("headline", f"{arm} {lab}", cm.get(key), "", f, f"{arm}/comfort/{key}")
            fair = cm.get("fairness") or {}
            for key in ("jain_appliance", "jain_ev", "jain_total", "jain_total_active"):
                t.add("headline", f"{arm} {key}", fair.get(key), "", f,
                      f"{arm}/comfort/fairness/{key}")
            usr = node.get("user") or {}
            # The bill totals are what Table 1 prints; without them the
            # manuscript's pounds column traces to nothing.
            t.add("headline", f"{arm} bill before", usr.get("total_cost_baseline_gbp"),
                  "GBP", f, f"{arm}/user/total_cost_baseline_gbp")
            t.add("headline", f"{arm} bill after", usr.get("total_cost_after_gbp"),
                  "GBP", f, f"{arm}/user/total_cost_after_gbp")
            t.add("headline", f"{arm} mean household bill saving",
                  usr.get("mean_household_saving_pct"), "%", f,
                  f"{arm}/user/mean_household_saving_pct")
            t.add("headline", f"{arm} weighted bill saving",
                  usr.get("weighted_total_saving_pct"), "%", f,
                  f"{arm}/user/weighted_total_saving_pct")

        dq = ms.get("data_quality") or {}
        rates = [v.get("nan_rate_pct") for v in dq.values() if isinstance(v, dict)]
        gaps = [v.get("max_gap_hours") for v in dq.values() if isinstance(v, dict)]
        rates = [r for r in rates if r is not None]
        gaps = [g for g in gaps if g is not None]
        if rates:
            t.add("data quality", "mean NaN rate", round(sum(rates) / len(rates), 3), "%", f,
                  "mean over data_quality/*/nan_rate_pct")
            t.add("data quality", "worst-house NaN rate", round(max(rates), 3), "%", f,
                  "max over data_quality/*/nan_rate_pct")
        if gaps:
            t.add("data quality", "longest gap", round(max(gaps), 2), "h", f,
                  "max over data_quality/*/max_gap_hours")

    # ---------- mechanism decomposition --------------------------------------
    md = _load("mechanism_decomposition.json")
    if md:
        f = "mechanism_decomposition.json"
        for arm, node in (md.get("rows") or {}).items():
            for lab, key, unit in [("peak", "peak_kw", "kW"), ("P95", "p95_kw", "kW"),
                                   ("peak reduction", "peak_red_pct", "%"),
                                   ("P95 reduction", "p95_red_pct", "%"),
                                   ("appliance decisions", "appliance_decisions", ""),
                                   ("EV decisions", "ev_decisions", "")]:
                t.add("mechanism", f"{arm} {lab}", node.get(key), unit, f, f"rows/{arm}/{key}")

    # ---------- multi-seed + bootstrap ---------------------------------------
    msd = _load("multiseed_results.json")
    if msd:
        f = "multiseed_results.json"
        t.add("multiseed", "seeds", len(msd.get("seeds") or []), "count", f, "seeds")
        for lvl, node in (msd.get("coordinated") or {}).items():
            for lab, key, unit in [("peak mean", "peak_kw_mean", "kW"),
                                   ("peak sd", "peak_kw_std", "kW"),
                                   ("P95 mean", "p95_kw_mean", "kW"),
                                   ("P95 sd", "p95_kw_std", "kW"),
                                   ("peak reduction mean", "peak_red_pct_mean", "%"),
                                   ("P95 reduction mean", "p95_red_pct_mean", "%")]:
                t.add("multiseed", f"{lvl} {lab}", node.get(key), unit, f,
                      f"coordinated/{lvl}/{key}")

    st = _load("stats_summary.json")
    if st:
        f = "stats_summary.json"
        for lvl, node in (st.get("accept_levels") or {}).items():
            for metric in ("peak_kw", "p95_kw"):
                sub = node.get(metric) or {}
                ci = sub.get("ci95")
                if ci:
                    t.add("bootstrap", f"{lvl} {metric} 95% CI",
                          f"[{ci[0]}, {ci[1]}]", "kW", f, f"accept_levels/{lvl}/{metric}/ci95")
            for key in ("peak_reduction_kw_boot_ci95", "p95_reduction_kw_boot_ci95"):
                ci = node.get(key)
                if ci:
                    t.add("bootstrap", f"{lvl} {key}", f"[{ci[0]}, {ci[1]}]", "kW", f,
                          f"accept_levels/{lvl}/{key}")

    # ---------- controller / strategy ladder ---------------------------------
    lad = _load("ev_strategy_ladder.json")
    if lad:
        f = "ev_strategy_ladder.json"
        for arm, node in (lad.get("rows") or {}).items():
            for lab, key, unit in [("peak", "peak_kw", "kW"), ("P95", "p95_kw", "kW"),
                                   ("peak reduction", "peak_red_pct", "%"),
                                   ("P95 reduction", "p95_red_pct", "%")]:
                t.add("ladder", f"{arm} {lab}", node.get(key), unit, f, f"rows/{arm}/{key}")

    mpc = _load("mpc_ladder.json")
    if mpc:
        f = "mpc_ladder.json"
        for arm in ("no_dr", "rule_85", "rule_100", "mpc_bound"):
            node = mpc.get(arm) or {}
            t.add("ladder", f"MPC ladder {arm} peak", node.get("peak_kw"), "kW", f, f"{arm}/peak_kw")
        t.add("ladder", "MPC LP energy conserved", mpc.get("energy_conserved"), "", f,
              "energy_conserved")
        nod, bound = (mpc.get("no_dr") or {}).get("peak_kw"), (mpc.get("mpc_bound") or {}).get("peak_kw")
        if nod and bound:
            span = round(nod - bound, 2)
            t.add("ladder", "max attainable peak reduction", span, "kW", f,
                  "derived: no_dr/peak_kw - mpc_bound/peak_kw")
            for arm in ("rule_85", "rule_100"):
                pk = (mpc.get(arm) or {}).get("peak_kw")
                if pk:
                    t.add("ladder", f"{arm} share of the bound",
                          round(100 * (nod - pk) / span, 1), "%", f,
                          f"derived: (no_dr - {arm})/(no_dr - mpc_bound)")

    # ---------- peak analysis: PAR and what sets the peak ---------------------
    pa = _load("peak_analysis.json")
    if pa:
        f = "peak_analysis.json"
        par = pa.get("par") or {}
        for arm in ("baseline", "independent", "coordinated"):
            t.add("peak", f"{arm} PAR", (par.get(arm) or {}).get("par"), "", f,
                  f"par/{arm}/par")
        t.add("peak", "PAR reduction (coordinated)", par.get("par_reduction_pct"), "%", f,
              "par/par_reduction_pct")
        ten = pa.get("ten_seed_peak") or {}
        for key, lab, unit in [("n_low_mode", "seeds in the low peak mode", ""),
                               ("low_min_kw", "low-mode peak min", "kW"),
                               ("low_max_kw", "low-mode peak max", "kW"),
                               ("n_high_mode", "seeds in the high peak mode", ""),
                               ("p95_sd_pct_of_mean", "P95 sd as % of mean", "%"),
                               ("peak_sd_pct_of_mean", "peak sd as % of mean", "%")]:
            t.add("peak", lab, ten.get(key), unit, f, f"ten_seed_peak/{key}")
        for i, v in enumerate(ten.get("high_values_kw") or []):
            t.add("peak", f"high-mode peak {i + 1}", v, "kW", f, f"ten_seed_peak/high_values_kw/{i}")
        for key, lab in [("low_reduction_pct_range", "low-mode peak reduction"),
                         ("high_reduction_pct_range", "high-mode peak reduction")]:
            lo, hi = ten.get(key) or (None, None)
            t.add("peak", f"{lab} (min)", lo, "%", f, f"ten_seed_peak/{key}/0")
            t.add("peak", f"{lab} (max)", hi, "%", f, f"ten_seed_peak/{key}/1")
        bs = pa.get("bound_share_pct") or {}
        t.add("peak", "full acceptance share of the bound", bs.get("full_acceptance"), "%", f,
              "bound_share_pct/full_acceptance")
        t.add("peak", "ten-seed mean share of the bound", bs.get("ten_seed_mean"), "%", f,
              "bound_share_pct/ten_seed_mean")
        t.add("peak", "reference run share of the bound", bs.get("reference_run"), "%", f,
              "bound_share_pct/reference_run")
        rn = pa.get("reference_night") or {}
        t.add("peak", "reference peak time", rn.get("peak_time"), "", f,
              "reference_night/peak_time")
        props = rn.get("proposals_that_night") or []
        t.add("peak", "reference night proposals rejected",
              sum(1 for p in props if not p.get("accepted")), "", f,
              "derived: count of reference_night/proposals_that_night with accepted=false")

    # ---------- EV factorial (external validity) ------------------------------
    fac = _load("ev_factorial.json")
    if fac:
        f = "ev_factorial.json"
        summ = fac.get("summary") or {}
        t.add("factorial", "scenarios", summ.get("n_scenarios"), "count", f, "summary/n_scenarios")
        t.add("factorial", "acceptance rate", summ.get("accept_p"), "", f, "summary/accept_p")
        for strat, node in (summ.get("strategies") or {}).items():
            for metric, unit in (("peak_red_pct", "%"), ("p95_red_pct", "%")):
                sub = node.get(metric) or {}
                t.add("factorial", f"{strat} {metric} mean", sub.get("mean"), unit, f,
                      f"summary/strategies/{strat}/{metric}/mean")
                t.add("factorial", f"{strat} {metric} sd", sub.get("sd"), unit, f,
                      f"summary/strategies/{strat}/{metric}/sd")
                t.add("factorial", f"{strat} {metric} min", sub.get("min"), unit, f,
                      f"summary/strategies/{strat}/{metric}/min")
                t.add("factorial", f"{strat} {metric} max", sub.get("max"), unit, f,
                      f"summary/strategies/{strat}/{metric}/max")
            t.add("factorial", f"{strat} scenarios with worse peak",
                  node.get("n_peak_worse"), "count", f,
                  f"summary/strategies/{strat}/n_peak_worse")
            t.add("factorial", f"{strat} scenarios with worse P95",
                  node.get("n_p95_worse"), "count", f,
                  f"summary/strategies/{strat}/n_p95_worse")
            t.add("factorial", f"{strat} scenarios with unchanged peak",
                  node.get("n_peak_unchanged"), "count", f,
                  f"summary/strategies/{strat}/n_peak_unchanged")
            unch, worse = node.get("n_peak_unchanged"), node.get("n_peak_worse")
            n_tot = summ.get("n_scenarios")
            if None not in (unch, worse, n_tot):
                t.add("factorial", f"{strat} scenarios with improved peak",
                      n_tot - unch - worse, "count", f,
                      f"derived: n_scenarios - n_peak_unchanged - n_peak_worse ({strat})")
        for arrival, node in (summ.get("by_arrival") or {}).items():
            for strat, sub in node.items():
                t.add("factorial", f"{arrival} arrivals, {strat}, peak reduction",
                      sub.get("peak_red_pct_mean"), "%", f,
                      f"summary/by_arrival/{arrival}/{strat}/peak_red_pct_mean")
                t.add("factorial", f"{arrival} arrivals, {strat}, P95 reduction",
                      sub.get("p95_red_pct_mean"), "%", f,
                      f"summary/by_arrival/{arrival}/{strat}/p95_red_pct_mean")
                t.add("factorial", f"{arrival} arrivals, {strat}, peak worse",
                      sub.get("n_peak_worse"), "count", f,
                      f"summary/by_arrival/{arrival}/{strat}/n_peak_worse")
                t.add("factorial", f"{arrival} arrivals, scenarios",
                      sub.get("n"), "count", f,
                      f"summary/by_arrival/{arrival}/{strat}/n")
        for ev_n, node in ((summ.get("by_factor") or {}).get("n_ev") or {}).items():
            for strat, sub in node.items():
                t.add("factorial", f"{ev_n} EVs/night, {strat}, P95 reduction",
                      sub.get("p95_red_pct_mean"), "%", f,
                      f"summary/by_factor/n_ev/{ev_n}/{strat}/p95_red_pct_mean")

    # ---------- LLM interface -------------------------------------------------
    for p in sorted(REPORTS.glob("llm_eval_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        summ = d.get("summary") or {}
        f = p.name
        for lab, key in [("model", "model"), ("events evaluated", "n_events"),
                         ("call completion rate", "call_completion_rate"),
                         ("transport failures", "n_transport_failures"),
                         ("schema-valid rate", "schema_valid_rate_of_completed"),
                         ("fully grounded message rate", "msg_all_citations_grounded_rate"),
                         ("ungrounded-number message rate", "msg_with_ungrounded_number_rate"),
                         ("unit-error message rate", "msg_with_unit_error_rate")]:
            t.add("llm", lab, summ.get(key), "", f, f"summary/{key}")
        lat = summ.get("latency_s") or {}
        for key in ("mean", "p50", "p95"):
            t.add("llm", f"latency {key}", lat.get(key), "s", f, f"summary/latency_s/{key}")

        # When the harness was run more than once, the single-run rate is not
        # the citable number — Ollama is not bit-reproducible across sessions,
        # so the paper must quote the range.
        env = d.get("server_env") or {}
        for lab, key in [("Ollama version", "ollama_version"),
                         ("model digest", "model_digest"),
                         ("model blob modified", "model_modified_at"),
                         ("quantization", "quantization"),
                         ("parameter size", "parameter_size")]:
            t.add("llm", lab, env.get(key), "", f, f"server_env/{key}")

        rep = d.get("repeatability") or {}
        t.add("llm", "repeats", rep.get("n_repeats"), "runs", f, "repeatability/n_repeats")
        for key, node in (rep.get("per_metric") or {}).items():
            t.add("llm", f"{key} over repeats",
                  f"{node.get('mean')} [{node.get('min')}, {node.get('max')}]", "",
                  f, f"repeatability/per_metric/{key}")

    # ---------- LLM output contract ------------------------------------------
    con = _load("llm_contract.json")
    if con:
        f = "llm_contract.json"
        t.add("llm contract", "messages", con.get("n_messages"), "count", f, "n_messages")
        for name, node in (con.get("per_check") or {}).items():
            if node.get("applicable"):
                t.add("llm contract", f"{name} pass rate", node.get("rate"), "", f,
                      f"per_check/{name}/rate")
                t.add("llm contract", f"{name} passed / applicable",
                      f"{node.get('passed')}/{node.get('applicable')}", "", f,
                      f"per_check/{name}")
        t.add("llm contract", "all checks passed", con.get("all_checks_passed"),
              "count", f, "all_checks_passed")
        t.add("llm contract", "all checks pass rate", con.get("all_checks_pass_rate"),
              "", f, "all_checks_pass_rate")

    # ---------- LLM cross-version comparison ---------------------------------
    cv = _load("llm_version_compare.json")
    if cv:
        f = "llm_version_compare.json"
        t.add("llm versions", "messages compared", cv.get("messages_compared"),
              "count", f, "messages_compared")
        t.add("llm versions", "messages with different text",
              cv.get("messages_with_different_text"), "count", f,
              "messages_with_different_text")
        t.add("llm versions", "messages with different text",
              cv.get("messages_with_different_text_pct"), "%", f,
              "messages_with_different_text_pct")
        for side in ("baseline", "current"):
            node = cv.get(side) or {}
            t.add("llm versions", f"{side} all-checks pass rate",
                  node.get("all_checks_pass_rate"), "", f,
                  f"{side}/all_checks_pass_rate")
            for name, sub in (node.get("per_check") or {}).items():
                t.add("llm versions", f"{side} {name}", sub.get("rate"), "", f,
                      f"{side}/per_check/{name}/rate")
        for name, delta in (cv.get("rate_change_current_minus_baseline") or {}).items():
            t.add("llm versions", f"{name} change across builds", delta, "", f,
                  f"rate_change_current_minus_baseline/{name}")
        for side, key in (("baseline", "baseline_server"), ("current", "current_server")):
            env = cv.get(key) or {}
            t.add("llm versions", f"{side} Ollama version", env.get("ollama_version"),
                  "", f, f"{key}/ollama_version")
            t.add("llm versions", f"{side} model digest", env.get("model_digest"),
                  "", f, f"{key}/model_digest")

    # ---------- forecast accuracy --------------------------------------------
    fe = _load("forecast_eval.json")
    if fe:
        f = "forecast_eval.json"
        for lab, key, unit in [("CNN-LSTM mean MAE", "mean_mae_lstm_w", "W"),
                               ("persistence mean MAE", "mean_mae_persistence_w", "W"),
                               ("CNN-LSTM mean RMSE", "mean_rmse_lstm_w", "W"),
                               ("persistence mean RMSE", "mean_rmse_persistence_w", "W"),
                               ("houses where CNN-LSTM wins", "lstm_wins_n_houses", "of 15"),
                               ("paired samples per house", "scored_steps_per_house", "10-min"),
                               ("lookback steps excluded", "lookback_steps_excluded", "10-min")]:
            t.add("forecast", lab, fe.get(key), unit, f, key)

    # ---------- sensitivity ---------------------------------------------------
    sen = _load("sensitivity_suite.json")
    if sen:
        f = "sensitivity_suite.json"
        for i, c in enumerate(sen.get("cohorts") or []):
            tag = c.get("tag")
            t.add("sensitivity", f"{tag} households", len(c.get("houses") or []), "count", f,
                  f"cohorts/{i}/houses")
            t.add("sensitivity", f"{tag} baseline peak",
                  (c.get("baseline") or {}).get("peak_kw"), "kW", f, f"cohorts/{i}/baseline/peak_kw")
            t.add("sensitivity", f"{tag} coordinated peak",
                  (c.get("coordinated") or {}).get("peak_kw"), "kW", f,
                  f"cohorts/{i}/coordinated/peak_kw")
            t.add("sensitivity", f"{tag} coordinated P95",
                  (c.get("coordinated") or {}).get("p95_kw"), "kW", f,
                  f"cohorts/{i}/coordinated/p95_kw")
            t.add("sensitivity", f"{tag} peak reduction", c.get("peak_red_pct"), "%", f,
                  f"cohorts/{i}/peak_red_pct")
            t.add("sensitivity", f"{tag} P95 reduction", c.get("p95_red_pct"), "%", f,
                  f"cohorts/{i}/p95_red_pct")
        for i, s in enumerate(sen.get("training_seeds") or []):
            co = s.get("coordinated") or {}
            t.add("sensitivity", f"{s.get('tag')} peak", co.get("peak_kw"), "kW", f,
                  f"training_seeds/{i}/coordinated/peak_kw")
            t.add("sensitivity", f"{s.get('tag')} P95", co.get("p95_kw"), "kW", f,
                  f"training_seeds/{i}/coordinated/p95_kw")

    # ---------- seasons -------------------------------------------------------
    sp = REPORTS / "season" / "season_summary.json"
    if sp.exists():
        try:
            rows = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            rows = []
        f = "season/season_summary.json"
        for i, s in enumerate(rows):
            tag = s.get("tag")
            t.add("season", f"{tag} window", " to ".join(s.get("window") or []), "", f,
                  f"{i}/window")
            t.add("season", f"{tag} test slots", s.get("test_slots"), "10-min", f, f"{i}/test_slots")
            t.add("season", f"{tag} baseline peak", (s.get("baseline") or {}).get("peak_kw"),
                  "kW", f, f"{i}/baseline/peak_kw")
            t.add("season", f"{tag} baseline P95", (s.get("baseline") or {}).get("p95_kw"),
                  "kW", f, f"{i}/baseline/p95_kw")
            t.add("season", f"{tag} coordinated peak", (s.get("coordinated") or {}).get("peak_kw"),
                  "kW", f, f"{i}/coordinated/peak_kw")
            t.add("season", f"{tag} coordinated P95", (s.get("coordinated") or {}).get("p95_kw"),
                  "kW", f, f"{i}/coordinated/p95_kw")
            t.add("season", f"{tag} peak reduction", s.get("peak_red_pct"), "%", f,
                  f"{i}/peak_red_pct")
            t.add("season", f"{tag} P95 reduction", s.get("p95_red_pct"), "%", f,
                  f"{i}/p95_red_pct")
            f12 = s.get("first_12d") or {}
            if f12:
                t.add("season", f"{tag} (first 12 d) baseline peak",
                      (f12.get("baseline") or {}).get("peak_kw"), "kW", f,
                      f"{i}/first_12d/baseline/peak_kw")
                t.add("season", f"{tag} (first 12 d) baseline P95",
                      (f12.get("baseline") or {}).get("p95_kw"), "kW", f,
                      f"{i}/first_12d/baseline/p95_kw")
                t.add("season", f"{tag} (first 12 d) coordinated peak",
                      (f12.get("coordinated") or {}).get("peak_kw"), "kW", f,
                      f"{i}/first_12d/coordinated/peak_kw")
                t.add("season", f"{tag} (first 12 d) coordinated P95",
                      (f12.get("coordinated") or {}).get("p95_kw"), "kW", f,
                      f"{i}/first_12d/coordinated/p95_kw")
                t.add("season", f"{tag} (first 12 d) peak reduction",
                      f12.get("peak_red_pct"), "%", f, f"{i}/first_12d/peak_red_pct")
                t.add("season", f"{tag} (first 12 d) P95 reduction",
                      f12.get("p95_red_pct"), "%", f, f"{i}/first_12d/p95_red_pct")

    # ---------- ablations + fairness -----------------------------------------
    ab = _load("ablation_results.json")
    if ab:
        f = "ablation_results.json"
        for group, rows in ab.items():
            if not isinstance(rows, list):
                continue
            for i, r in enumerate(rows):
                tag = r.get("tag")
                t.add("ablation", f"{group}: {tag} peak", r.get("peak_kw"), "kW", f,
                      f"{group}/{i}/peak_kw")
                t.add("ablation", f"{group}: {tag} P95 reduction", r.get("p95_reduction"), "%", f,
                      f"{group}/{i}/p95_reduction")
                t.add("ablation", f"{group}: {tag} decisions", r.get("total_recs"), "count", f,
                      f"{group}/{i}/total_recs")

    fw = _load("fairness_sweep.json")
    if isinstance(fw, list):
        f = "fairness_sweep.json"
        for i, r in enumerate(fw):
            tag = r.get("tag")
            t.add("fairness", f"{tag} Jain (total)", r.get("fairness"), "", f, f"{i}/fairness")
            t.add("fairness", f"{tag} Jain (appliance)", r.get("fairness_appliance"), "", f,
                  f"{i}/fairness_appliance")
            t.add("fairness", f"{tag} P95 reduction", r.get("p95_reduction"), "%", f,
                  f"{i}/p95_reduction")
            t.add("fairness", f"{tag} decisions skipped", r.get("n_skipped_by_fairness"),
                  "count", f, f"{i}/n_skipped_by_fairness")

    return t.rows


def main() -> None:
    rows = collect()
    if not rows:
        print("no artifacts found — run regen_all first")
        return

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "paper_numbers.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Paper numbers — generated file, do not hand-edit",
        "",
        "Every value below was read from a result file at generation time. Cite these;",
        "do not retype a number from an earlier draft of the manuscript.",
        "",
        "Regenerate with `python -m multi_household.experiments.paper_numbers`.",
        "",
    ]
    order: list[str] = []
    for r in rows:
        if r["section"] not in order:
            order.append(r["section"])
    for sec in order:
        lines += [f"## {sec}", "",
                  "| Quantity | Value | Unit | Source file | JSON path |",
                  "|---|---|---|---|---|"]
        for r in rows:
            if r["section"] == sec:
                lines.append(f"| {r['label']} | {r['value']} | {r['unit']} | "
                             f"`{r['source']}` | `{r['path']}` |")
        lines.append("")

    out = REPORTS / "paper_numbers.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {out}  ({len(rows)} values across {len(order)} sections)")
    for sec in order:
        print(f"  {sec:<14} {sum(1 for r in rows if r['section'] == sec):>4}")


if __name__ == "__main__":
    main()
