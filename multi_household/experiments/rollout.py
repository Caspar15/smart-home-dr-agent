"""End-to-end multi-household rollout over the test period.

Three rollout modes for comparison:
  baseline      : no DR — served = demand. Sanity ground truth.
  independent   : each house reacts to fixed ToU only. No aggregator-side
                  peak event. Useful for measuring rebound peak risk.
  coordinated   : full system — aggregator broadcasts dynamic price +
                  peak_event flag. Houses defer appliance cycles.

Run:
   python -m multi_household.experiments.rollout \\
      --houses 1 2 3 5 7 9 15 20 \\
      --days 14 --mode coordinated --user-accept 0.85
"""
from __future__ import annotations
import argparse, time, json
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from multi_household.config import (
    CLEAN_HOUSES, PEAK_HOURS_LOCAL, OFFPEAK_HOURS,
    PEAK_PRICE_GBP, MID_PRICE_GBP, OFFPEAK_PRICE_GBP,
    GRID_THRESHOLD_W, RESAMPLE_FREQ,
)
from multi_household.data.preprocess import prepare_house
from multi_household.data.appliance_map import HOUSE_APPLIANCES, classify
from multi_household.forecasting.per_house_lstm import (
    CNNLSTM, load_forecaster, MODEL_DIR,
)
from multi_household.aggregator.price_broadcast import (
    aggregate_and_price, base_tou_price, is_off_peak, Broadcast,
)
from multi_household.aggregator.ev_coordinator import advisory_ev_schedule
from multi_household.agent.appliance_controller import (
    HouseAgentState, decide_step, build_state, _hour_bucket,
)
from multi_household.llm.advisor import template_recommendation, Recommendation


REPORTS = Path(__file__).resolve().parents[2] / "reports" / "multi_household"
REPORTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Closed-loop helper — load user-rejection patterns from saved choices
# ---------------------------------------------------------------------------

def load_user_rejection_rates(house_id: int) -> dict[str, float]:
    """Read user_choices.json and compute per-(appliance, hour_bucket) reject
    rate for one house. Returns {} if file missing or empty.

    Pattern key format: "{appliance_short}@{bucket}",
        e.g. "washing_machine@peak"  →  rate in [0.0, 1.0]
    """
    path = REPORTS / "user_choices.json"
    if not path.exists():
        return {}
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    house_entries = [e for e in log if e.get("house") == house_id]
    if not house_entries:
        return {}

    pattern_total: dict[str, int] = {}
    pattern_reject: dict[str, int] = {}
    for e in house_entries:
        apl  = e.get("rec_appliance", "") or ""
        step = e.get("rec_step", -1)
        if not apl or step < 0:
            continue
        hour = (step % 144) // 6           # 144 steps/day, 6 steps/hour
        key  = f"{apl}@{_hour_bucket(hour)}"
        pattern_total[key] = pattern_total.get(key, 0) + 1
        if e.get("user_choice") == 2:
            pattern_reject[key] = pattern_reject.get(key, 0) + 1

    rates: dict[str, float] = {}
    for k, total in pattern_total.items():
        if total >= 2:                     # need ≥ 2 samples to trust
            rates[k] = pattern_reject.get(k, 0) / total
    return rates


# ---------------------------------------------------------------------------
# 1. Pre-pass: run each frozen LSTM over the test set once
# ---------------------------------------------------------------------------

def compute_all_forecasts(houses: list[int],
                          n_test_steps: int | None = None,
                          device: str | None = None) -> dict:
    """Run each house's frozen LSTM over its full test set.

    Returns:
        {
          house_id: {
             "test_df":     pd.DataFrame  (length T) (with appliance cols),
             "forecast_w":  np.ndarray (T,) — LSTM ŷ(t+1), aligned with test_df.iloc[t]
             "deferable_cols": list[str],
             "appliance_cols": list[str],
          }
        }
    Only the first n_test_steps are used (per house) if given.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out = {}
    for h in houses:
        data = prepare_house(h)
        test_df = data["test_df"]
        fcols   = data["feature_cols"]
        target  = data["target_col"]

        # find appliance columns
        deferable_cols = [c for c in test_df.columns
                          if c.startswith("appliance_")
                          and "_w" in c
                          and classify(_col_name_to_appliance(c, h)) == "deferable"]
        appliance_cols = [c for c in test_df.columns
                          if c.startswith("appliance_") and "_w" in c]

        model, ckpt = load_forecaster(h, device=device)
        lookback = ckpt["lookback"]

        # Standardize features using saved mean/scale
        fx_mean  = np.array(ckpt["fx_mean"],  dtype=np.float32)
        fx_scale = np.array(ckpt["fx_scale"], dtype=np.float32)
        ty_mean  = float(ckpt["ty_mean"])
        ty_scale = float(ckpt["ty_scale"])

        Xte = ((test_df[fcols].values - fx_mean) / fx_scale).astype(np.float32)
        T = len(Xte)

        forecasts = np.zeros(T, dtype=np.float32)
        # First `lookback` steps: not enough history to use the model;
        # fall back to persistence (use current demand as the forecast).
        forecasts[:lookback] = test_df[target].values[:lookback].astype(np.float32)

        # Batched windowed prediction
        batch = 1024
        with torch.no_grad():
            for start in range(lookback, T, batch):
                end = min(T, start + batch)
                # Build window for each row [start, end)
                wins = np.stack([Xte[i-lookback:i] for i in range(start, end)])
                x = torch.from_numpy(wins).to(device)
                pred_scaled = model(x).cpu().numpy()
                forecasts[start:end] = pred_scaled * ty_scale + ty_mean

        if n_test_steps is not None:
            test_df    = test_df.iloc[:n_test_steps].reset_index(drop=True)
            forecasts  = forecasts[:n_test_steps]

        out[h] = {
            "test_df":         test_df,
            "forecast_w":      forecasts,
            "deferable_cols":  deferable_cols,
            "appliance_cols":  appliance_cols,
        }
        print(f"  H{h:02d}: T={len(test_df)} steps, "
              f"deferable={len(deferable_cols)} cols")
    return out


def _col_name_to_appliance(col: str, house_id: int) -> str:
    """Reverse-lookup: given e.g. 'appliance_washing_machine_w', find the
    human name from HOUSE_APPLIANCES[house_id]."""
    base = col.replace("appliance_", "").replace("_w", "")
    base = base.replace("_ch", " ch")  # for House 4 duplicate
    for ch, name in HOUSE_APPLIANCES[house_id].items():
        from multi_household.data.refit_loader import _safe_name
        if _safe_name(name) == base.replace(" ", "_"):
            return name
        if _safe_name(name) in base:
            return name
    return base


# ---------------------------------------------------------------------------
# 2. The actual rollout loop
# ---------------------------------------------------------------------------

EV_COL = "appliance_synthetic_ev_w"


def rollout(houses_data: dict,
            mode: str = "coordinated",
            user_accept: float = 1.0,
            verbose: bool = True,
            forecast_mode: str = "lstm",
            ev_smart: bool = False,
            ev_seed: int | None = None,
            closed_loop: bool = True,
            rejection_override: dict[str, float] | None = None) -> dict:
    """Step through the test period for all houses simultaneously.

    Returns a dict with:
      served_w[h]:        np.ndarray   per-house actual served Wh after DR
      demand_w[h]:        np.ndarray   per-house real demand (= no-DR baseline)
      broadcasts:         list[Broadcast]
      recommendations:    list[Recommendation]
      agent_state[h]:     HouseAgentState
      timestamps:         np.ndarray of pd.Timestamp
    """
    assert mode in {"baseline", "independent", "coordinated"}

    houses = sorted(houses_data.keys())
    T = min(len(houses_data[h]["test_df"]) for h in houses)
    if verbose:
        print(f"  rolling out {len(houses)} houses × {T} steps, mode={mode}")

    served_w = {h: np.zeros(T, dtype=np.float32) for h in houses}
    demand_w = {h: houses_data[h]["test_df"]["aggregate_w"].values[:T].astype(np.float32) for h in houses}
    # Forecast source: LSTM (saved per-house) or persistence (ŷ(t+1) = y(t))
    if forecast_mode == "lstm":
        forecast_w = {h: houses_data[h]["forecast_w"][:T].astype(np.float32) for h in houses}
    elif forecast_mode == "persistence":
        forecast_w = {h: np.roll(demand_w[h], 1).astype(np.float32) for h in houses}
        for h in houses:                           # first step has no prior
            forecast_w[h][0] = demand_w[h][0]
    else:
        raise ValueError(f"unknown forecast_mode {forecast_mode}")
    timestamps = pd.to_datetime(houses_data[houses[0]]["test_df"]["time"].values[:T])
    # Pre-extract appliance load matrices once
    appliance_loads = {
        h: {col: houses_data[h]["test_df"][col].values[:T].astype(np.float32)
            for col in houses_data[h]["appliance_cols"]}
        for h in houses
    }

    # ── EV smart charging (coordinated only) ──────────────────────────────
    # ADVISORY: each night's EV reschedule is a recommendation the user accepts
    # (prob = user_accept) or rejects (EV stays at its natural time). So user
    # acceptance drives the peak — accept 0 leaves the pile-up, accept 1 fully
    # staggers it. EV is taken out of the per-house agent's hands either way.
    # ev_orig/ev_shift are populated only for ACCEPTED blocks (zero elsewhere).
    ev_orig  = {h: np.zeros(T, dtype=np.float32) for h in houses}
    ev_shift = {h: np.zeros(T, dtype=np.float32) for h in houses}
    if ev_smart:
        ev_houses = {h: appliance_loads[h][EV_COL][:T]
                     for h in houses if EV_COL in appliance_loads[h]}
        if ev_houses:
            # ev_seed=None keeps the fixed default (reproducible headline);
            # multi-seed runs pass a varying seed so the nightly accept
            # decisions contribute to the error bars.
            ev_kwargs = {} if ev_seed is None else {"seed": ev_seed}
            oa, sa, (ev_reco, ev_acc) = advisory_ev_schedule(
                ev_houses, timestamps, accept_rate=user_accept, **ev_kwargs)
            ev_orig.update(oa)
            ev_shift.update(sa)
            if verbose:
                print(f"  EV advisory: {ev_acc}/{ev_reco} nightly reschedules "
                      f"accepted (accept_rate={user_accept})")

    def _agent_deferable(h):
        cols = houses_data[h]["deferable_cols"]
        if ev_smart:
            cols = [c for c in cols if c != EV_COL]   # coordinator owns the EV
        return cols

    # Build agent state per house — ★ inject closed-loop rejection rates
    # (closed_loop=False skips the user-history injection: the on/off ablation)
    agents = {h: build_state(h, _agent_deferable(h)) for h in houses}
    n_loaded = 0
    if closed_loop:
        for h in houses:
            rates = load_user_rejection_rates(h)
            if rates:
                agents[h].pattern_rejection_rate = rates
                n_loaded += len(rates)
        # Synthetic override for the closed-loop STRESS ablation: pretend every
        # user rejected these patterns, so suppression provably engages in-rollout.
        if rejection_override:
            for h in houses:
                agents[h].pattern_rejection_rate.update(rejection_override)
    if verbose and n_loaded > 0:
        print(f"  closed loop: loaded {n_loaded} rejection patterns from user_choices.json")

    broadcasts: list[Broadcast] = []
    recs: list[Recommendation] = []
    prev_agg_served = 0.0                          # for aggregator feedback
    # Energy-conservation guard: how much energy each house has actually had
    # removed from served (Wh). Releases are capped at this so served can never
    # exceed demand in total (no energy created by data inconsistencies).
    banked_wh = {h: 0.0 for h in houses}

    for t in range(T):
        hour = int(timestamps[t].hour)

        # Step 1: aggregator
        if mode == "coordinated":
            forecasts = [forecast_w[h][t] for h in houses]
            bc = aggregate_and_price(forecasts, timestep=t, hour=hour,
                                     prev_served_w=prev_agg_served)
        elif mode == "independent":
            # Independent: every house reacts to the FIXED ToU price.
            # peak_event is on whenever ToU is in the peak tier (17:00-22:00).
            # This is the "no aggregator coordination" baseline — all houses
            # defer in lockstep during peak hours, creating a rebound peak.
            is_peak_hour = (PEAK_HOURS_LOCAL[0] <= hour < PEAK_HOURS_LOCAL[1])
            bc = Broadcast(
                timestep=t, hour=hour,
                p_now_gbp_kwh=base_tou_price(hour),
                p_off_gbp_kwh=OFFPEAK_PRICE_GBP,
                peak_event=is_peak_hour,
                aggregate_forecast_w=float(sum(forecast_w[h][t] for h in houses)),
                threshold_w=GRID_THRESHOLD_W,
                overage_ratio=0.0,
            )
        else:
            # baseline: no DR at all
            bc = Broadcast(
                timestep=t, hour=hour,
                p_now_gbp_kwh=base_tou_price(hour),
                p_off_gbp_kwh=OFFPEAK_PRICE_GBP,
                peak_event=False,
                aggregate_forecast_w=float(sum(forecast_w[h][t] for h in houses)),
                threshold_w=GRID_THRESHOLD_W,
                overage_ratio=0.0,
            )
        broadcasts.append(bc)

        # Step 2: each house decides
        for h in houses:
            current_apps = {col: float(appliance_loads[h][col][t])
                            for col in appliance_loads[h]}

            if mode == "baseline":
                # No DR — served = demand
                served_w[h][t] = demand_w[h][t]
                continue

            decision = decide_step(
                state=agents[h],
                appliance_loads_w=current_apps,
                forecast_w=float(forecast_w[h][t]),
                broadcast=bc,
                step=t,
                accept_rate=user_accept,
                current_demand_w=float(demand_w[h][t]),
            )

            # Apply decision to served load. The agent now always reports a
            # per-STEP energy (deferred_wh / released_wh) for this 10-min step,
            # so we just multiply by 6 to convert Wh → W. Cap subtraction at
            # current demand so served can never go negative.
            # Start from real demand, but with the EV moved to its coordinated
            # (staggered) slot: remove the original EV block, add the shifted one.
            served = float(demand_w[h][t]) - float(ev_orig[h][t]) + float(ev_shift[h][t])
            if decision.action == "defer":
                sub_w = min(decision.deferred_wh * 6.0, served)
                served = served - sub_w
                banked_wh[h] += sub_w / 6.0          # credit only what we removed
            elif decision.action == "release":
                # never release more than this house has actually banked
                rel_w = min(decision.released_wh * 6.0, banked_wh[h] * 6.0)
                served = served + rel_w
                banked_wh[h] -= rel_w / 6.0
            served_w[h][t] = max(served, 0.0)

            # Step 3: LLM template recommendation (only meaningful events)
            rec = template_recommendation(h, t, decision, bc)
            if rec is not None:
                recs.append(rec)

        # Step 4: feedback for aggregator next step
        prev_agg_served = float(sum(served_w[h][t] for h in houses))

    if verbose:
        n_suppressed = sum(agents[h].n_suppressed_by_user_history for h in houses)
        print(f"  done. {len(recs)} recommendations across all houses.")
        if n_suppressed > 0:
            print(f"        suppressed {n_suppressed} would-be recs based on user history.")

    return {
        "served_w":         served_w,
        "demand_w":         demand_w,
        "forecast_w":       forecast_w,
        "broadcasts":       broadcasts,
        "recommendations":  recs,
        "agent_state":      agents,
        "timestamps":       timestamps,
        "houses":           houses,
        "mode":             mode,
    }


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", nargs="+", type=int, default=CLEAN_HOUSES)
    ap.add_argument("--days", type=int, default=14,
                    help="how many days of the test set to roll out (None = all)")
    ap.add_argument("--mode", default="coordinated",
                    choices=["baseline", "independent", "coordinated", "all"])
    ap.add_argument("--user-accept", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--forecast-mode", default="lstm",
                    choices=["lstm", "persistence"])
    ap.add_argument("--ev-smart", action=argparse.BooleanOptionalAction, default=True,
                    help="stagger the synthetic EVs across the overnight trough "
                         "(coordinated mode only; --no-ev-smart to ablate)")
    args = ap.parse_args()

    import random; random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_test_steps = args.days * 144 if args.days else None

    print(f"\n[1/3] Loading data + running per-house LSTM over test set ...")
    t0 = time.time()
    houses_data = compute_all_forecasts(args.houses, n_test_steps=n_test_steps)
    print(f"      elapsed {time.time()-t0:.0f}s")

    modes = ["baseline", "independent", "coordinated"] if args.mode == "all" else [args.mode]

    results = {}
    for mode in modes:
        print(f"\n[2/3] Rollout (mode={mode}, forecast={args.forecast_mode}) ...")
        t0 = time.time()
        # EV stagger is a coordination action → only in coordinated mode.
        results[mode] = rollout(houses_data, mode=mode,
                                user_accept=args.user_accept,
                                forecast_mode=args.forecast_mode,
                                ev_smart=(args.ev_smart and mode == "coordinated"))
        print(f"      elapsed {time.time()-t0:.0f}s")

    # Save raw results
    out_dir = REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode, r in results.items():
        path = out_dir / f"rollout_{mode}.npz"
        np.savez(path,
                 served=np.stack([r["served_w"][h] for h in r["houses"]]),
                 demand=np.stack([r["demand_w"][h] for h in r["houses"]]),
                 forecast=np.stack([r["forecast_w"][h] for h in r["houses"]]),
                 timestamps=r["timestamps"].astype("datetime64[s]"),
                 houses=np.array(r["houses"]))
        print(f"  saved {path}")

        # Save defer wait log (per house): every (wait_steps, energy_wh) record
        wait_path = out_dir / f"rollout_{mode}_waitlog.json"
        wait_logs = {int(h): r["agent_state"][h].wait_log for h in r["houses"]}
        wait_path.write_text(json.dumps(wait_logs, ensure_ascii=False, indent=2),
                             encoding="utf-8")

        # Dump recommendations as JSON
        recs_path = out_dir / f"rollout_{mode}_recs.json"
        recs_path.write_text(json.dumps([asdict(rr) for rr in r["recommendations"]],
                                         ensure_ascii=False, indent=2),
                             encoding="utf-8")

    # Print quick stats
    print(f"\n[3/3] Summary")
    for mode, r in results.items():
        total_demand = float(sum(r["demand_w"][h].sum() for h in r["houses"]))
        total_served = float(sum(r["served_w"][h].sum() for h in r["houses"]))
        agg = np.stack([r["served_w"][h] for h in r["houses"]]).sum(axis=0)
        p95 = float(np.percentile(agg, 95))
        peak = float(agg.max())
        n_rec = len(r["recommendations"])
        print(f"  [{mode:12s}] demand={total_demand/1e6:.2f} MWh, "
              f"served={total_served/1e6:.2f} MWh, "
              f"agg P95={p95/1000:.2f} kW, "
              f"peak={peak/1000:.2f} kW, "
              f"recs={n_rec}")


if __name__ == "__main__":
    main()
