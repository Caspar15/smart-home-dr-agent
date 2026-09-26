# Paper numbers — generated file, do not hand-edit

Every value below was read from a result file at generation time. Cite these;
do not retype a number from an earlier draft of the manuscript.

Regenerate with `python -m multi_household.experiments.paper_numbers`.

## provenance

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| run timestamp (UTC) | 2026-09-24T08:57:17Z |  | `run_manifest.json` | `run_utc` |
| git commit | 32b1bb4006d22067ef443c0b20e24311e5e701b2 |  | `run_manifest.json` | `git_commit` |
| households | 15 | count | `run_manifest.json` | `houses` |
| household IDs | 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 20 |  | `run_manifest.json` | `houses` |
| random seed | 42 |  | `run_manifest.json` | `seed` |
| EV seed | 20260710 |  | `run_manifest.json` | `ev_seed` |
| acceptance rate | 0.85 |  | `run_manifest.json` | `user_accept` |
| forecast mode | lstm |  | `run_manifest.json` | `forecast_mode` |
| EV coordinator enabled | True |  | `run_manifest.json` | `ev_smart` |
| grid threshold G | 17712.6 | W | `run_manifest.json` | `grid_threshold_w` |
| test-window steps | 2016 | 10-min | `run_manifest.json` | `test_steps` |
| test-window length | 14.0 | days | `run_manifest.json` | `test_days` |
| train/test split | 2014-06-30 12:00 |  | `run_manifest.json` | `split_at` |
| clean window | 2014-04-30 to 2014-07-14 12:00 |  | `run_manifest.json` | `clean_window` |
| python version | 3.12.7 |  | `run_manifest.json` | `versions/python` |
| torch version | 2.5.1+cu121 |  | `run_manifest.json` | `versions/torch` |
| numpy version | 1.26.4 |  | `run_manifest.json` | `versions/numpy` |
| pandas version | 2.2.2 |  | `run_manifest.json` | `versions/pandas` |

## headline

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| baseline peak | 40.37 | kW | `metrics_summary.json` | `baseline/grid/agg_served_peak_kw` |
| baseline P95 | 26.79 | kW | `metrics_summary.json` | `baseline/grid/agg_served_p95_kw` |
| baseline P95 reduction | 0.0 | % | `metrics_summary.json` | `baseline/grid/p95_reduction_pct` |
| baseline peak reduction | 0.0 | % | `metrics_summary.json` | `derived from baseline/grid/agg_served_peak_kw vs baseline` |
| baseline energy drift | 0.0 | % | `metrics_summary.json` | `baseline/energy_conservation/diff_pct` |
| baseline terminal buffer debt | 0.0 | kWh | `metrics_summary.json` | `baseline/energy_conservation/terminal_buffer_debt_kwh` |
| baseline bill before | 540.39 | GBP | `metrics_summary.json` | `baseline/user/total_cost_baseline_gbp` |
| baseline bill after | 540.39 | GBP | `metrics_summary.json` | `baseline/user/total_cost_after_gbp` |
| baseline mean household bill saving | 0.0 | % | `metrics_summary.json` | `baseline/user/mean_household_saving_pct` |
| baseline weighted bill saving | 0.0 | % | `metrics_summary.json` | `baseline/user/weighted_total_saving_pct` |
| independent peak | 34.91 | kW | `metrics_summary.json` | `independent/grid/agg_served_peak_kw` |
| independent P95 | 24.73 | kW | `metrics_summary.json` | `independent/grid/agg_served_p95_kw` |
| independent P95 reduction | 7.71 | % | `metrics_summary.json` | `independent/grid/p95_reduction_pct` |
| independent peak reduction | 13.52 | % | `metrics_summary.json` | `derived from independent/grid/agg_served_peak_kw vs baseline` |
| independent energy drift | -0.062 | % | `metrics_summary.json` | `independent/energy_conservation/diff_pct` |
| independent terminal buffer debt | 2.139 | kWh | `metrics_summary.json` | `independent/energy_conservation/terminal_buffer_debt_kwh` |
| independent appliance decisions | 162 |  | `metrics_summary.json` | `independent/comfort/appliance_decisions` |
| independent appliance accepted | 138 |  | `metrics_summary.json` | `independent/comfort/appliance_accepted` |
| independent appliance accept rate | 0.852 |  | `metrics_summary.json` | `independent/comfort/appliance_accept_rate` |
| independent EV decisions | 0 |  | `metrics_summary.json` | `independent/comfort/ev_decisions` |
| independent EV accepted | 0 |  | `metrics_summary.json` | `independent/comfort/ev_accepted` |
| independent total decisions | 162 |  | `metrics_summary.json` | `independent/comfort/total_decisions` |
| independent total accepted | 138 |  | `metrics_summary.json` | `independent/comfort/total_accepted` |
| independent surface messages | 2072 |  | `metrics_summary.json` | `independent/comfort/surface_messages_total` |
| independent jain_appliance | 0.6358 |  | `metrics_summary.json` | `independent/comfort/fairness/jain_appliance` |
| independent jain_total | 0.6358 |  | `metrics_summary.json` | `independent/comfort/fairness/jain_total` |
| independent jain_total_active | 0.6358 |  | `metrics_summary.json` | `independent/comfort/fairness/jain_total_active` |
| independent bill before | 540.39 | GBP | `metrics_summary.json` | `independent/user/total_cost_baseline_gbp` |
| independent bill after | 523.22 | GBP | `metrics_summary.json` | `independent/user/total_cost_after_gbp` |
| independent mean household bill saving | 2.26 | % | `metrics_summary.json` | `independent/user/mean_household_saving_pct` |
| independent weighted bill saving | 3.18 | % | `metrics_summary.json` | `independent/user/weighted_total_saving_pct` |
| coordinated peak | 32.41 | kW | `metrics_summary.json` | `coordinated/grid/agg_served_peak_kw` |
| coordinated P95 | 18.65 | kW | `metrics_summary.json` | `coordinated/grid/agg_served_p95_kw` |
| coordinated P95 reduction | 30.39 | % | `metrics_summary.json` | `coordinated/grid/p95_reduction_pct` |
| coordinated peak reduction | 19.72 | % | `metrics_summary.json` | `derived from coordinated/grid/agg_served_peak_kw vs baseline` |
| coordinated energy drift | -0.051 | % | `metrics_summary.json` | `coordinated/energy_conservation/diff_pct` |
| coordinated terminal buffer debt | 1.775 | kWh | `metrics_summary.json` | `coordinated/energy_conservation/terminal_buffer_debt_kwh` |
| coordinated appliance decisions | 80 |  | `metrics_summary.json` | `coordinated/comfort/appliance_decisions` |
| coordinated appliance accepted | 71 |  | `metrics_summary.json` | `coordinated/comfort/appliance_accepted` |
| coordinated appliance accept rate | 0.887 |  | `metrics_summary.json` | `coordinated/comfort/appliance_accept_rate` |
| coordinated EV decisions | 44 |  | `metrics_summary.json` | `coordinated/comfort/ev_decisions` |
| coordinated EV accepted | 39 |  | `metrics_summary.json` | `coordinated/comfort/ev_accepted` |
| coordinated EV accept rate | 0.886 |  | `metrics_summary.json` | `coordinated/comfort/ev_accept_rate` |
| coordinated total decisions | 124 |  | `metrics_summary.json` | `coordinated/comfort/total_decisions` |
| coordinated total accepted | 110 |  | `metrics_summary.json` | `coordinated/comfort/total_accepted` |
| coordinated surface messages | 821 |  | `metrics_summary.json` | `coordinated/comfort/surface_messages_total` |
| coordinated jain_appliance | 0.5116 |  | `metrics_summary.json` | `coordinated/comfort/fairness/jain_appliance` |
| coordinated jain_ev | 0.9878 |  | `metrics_summary.json` | `coordinated/comfort/fairness/jain_ev` |
| coordinated jain_total | 0.6351 |  | `metrics_summary.json` | `coordinated/comfort/fairness/jain_total` |
| coordinated jain_total_active | 0.6805 |  | `metrics_summary.json` | `coordinated/comfort/fairness/jain_total_active` |
| coordinated bill before | 540.39 | GBP | `metrics_summary.json` | `coordinated/user/total_cost_baseline_gbp` |
| coordinated bill after | 539.81 | GBP | `metrics_summary.json` | `coordinated/user/total_cost_after_gbp` |
| coordinated mean household bill saving | 0.15 | % | `metrics_summary.json` | `coordinated/user/mean_household_saving_pct` |
| coordinated weighted bill saving | 0.11 | % | `metrics_summary.json` | `coordinated/user/weighted_total_saving_pct` |

## data quality

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| mean NaN rate | 0.495 | % | `metrics_summary.json` | `mean over data_quality/*/nan_rate_pct` |
| worst-house NaN rate | 1.205 | % | `metrics_summary.json` | `max over data_quality/*/nan_rate_pct` |
| longest gap | 5.7 | h | `metrics_summary.json` | `max over data_quality/*/max_gap_hours` |

## mechanism

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| natural peak | 25.82 | kW | `mechanism_decomposition.json` | `rows/natural/peak_kw` |
| natural P95 | 11.62 | kW | `mechanism_decomposition.json` | `rows/natural/p95_kw` |
| ev_no_dr peak | 40.37 | kW | `mechanism_decomposition.json` | `rows/ev_no_dr/peak_kw` |
| ev_no_dr P95 | 26.79 | kW | `mechanism_decomposition.json` | `rows/ev_no_dr/p95_kw` |
| ev_no_dr peak reduction | 0.0 | % | `mechanism_decomposition.json` | `rows/ev_no_dr/peak_red_pct` |
| ev_no_dr P95 reduction | 0.0 | % | `mechanism_decomposition.json` | `rows/ev_no_dr/p95_red_pct` |
| agent_with_ev peak | 33.33 | kW | `mechanism_decomposition.json` | `rows/agent_with_ev/peak_kw` |
| agent_with_ev P95 | 24.61 | kW | `mechanism_decomposition.json` | `rows/agent_with_ev/p95_kw` |
| agent_with_ev peak reduction | 17.4 | % | `mechanism_decomposition.json` | `rows/agent_with_ev/peak_red_pct` |
| agent_with_ev P95 reduction | 8.1 | % | `mechanism_decomposition.json` | `rows/agent_with_ev/p95_red_pct` |
| agent_with_ev appliance decisions | 92 |  | `mechanism_decomposition.json` | `rows/agent_with_ev/appliance_decisions` |
| agent_with_ev EV decisions | 0 |  | `mechanism_decomposition.json` | `rows/agent_with_ev/ev_decisions` |
| agent_no_ev peak | 40.33 | kW | `mechanism_decomposition.json` | `rows/agent_no_ev/peak_kw` |
| agent_no_ev P95 | 26.82 | kW | `mechanism_decomposition.json` | `rows/agent_no_ev/p95_kw` |
| agent_no_ev peak reduction | 0.1 | % | `mechanism_decomposition.json` | `rows/agent_no_ev/peak_red_pct` |
| agent_no_ev P95 reduction | -0.1 | % | `mechanism_decomposition.json` | `rows/agent_no_ev/p95_red_pct` |
| agent_no_ev appliance decisions | 80 |  | `mechanism_decomposition.json` | `rows/agent_no_ev/appliance_decisions` |
| agent_no_ev EV decisions | 0 |  | `mechanism_decomposition.json` | `rows/agent_no_ev/ev_decisions` |
| coord_only peak | 32.41 | kW | `mechanism_decomposition.json` | `rows/coord_only/peak_kw` |
| coord_only P95 | 18.58 | kW | `mechanism_decomposition.json` | `rows/coord_only/p95_kw` |
| coord_only peak reduction | 19.7 | % | `mechanism_decomposition.json` | `rows/coord_only/peak_red_pct` |
| coord_only P95 reduction | 30.6 | % | `mechanism_decomposition.json` | `rows/coord_only/p95_red_pct` |
| coord_only appliance decisions | 0 |  | `mechanism_decomposition.json` | `rows/coord_only/appliance_decisions` |
| coord_only EV decisions | 44 |  | `mechanism_decomposition.json` | `rows/coord_only/ev_decisions` |
| full peak | 32.41 | kW | `mechanism_decomposition.json` | `rows/full/peak_kw` |
| full P95 | 18.65 | kW | `mechanism_decomposition.json` | `rows/full/p95_kw` |
| full peak reduction | 19.7 | % | `mechanism_decomposition.json` | `rows/full/peak_red_pct` |
| full P95 reduction | 30.4 | % | `mechanism_decomposition.json` | `rows/full/p95_red_pct` |
| full appliance decisions | 80 |  | `mechanism_decomposition.json` | `rows/full/appliance_decisions` |
| full EV decisions | 44 |  | `mechanism_decomposition.json` | `rows/full/ev_decisions` |

## multiseed

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| seeds | 10 | count | `multiseed_results.json` | `seeds` |
| accept=0.00 peak mean | 40.37 | kW | `multiseed_results.json` | `coordinated/accept=0.00/peak_kw_mean` |
| accept=0.00 peak sd | 0.0 | kW | `multiseed_results.json` | `coordinated/accept=0.00/peak_kw_std` |
| accept=0.00 P95 mean | 26.79 | kW | `multiseed_results.json` | `coordinated/accept=0.00/p95_kw_mean` |
| accept=0.00 P95 sd | 0.0 | kW | `multiseed_results.json` | `coordinated/accept=0.00/p95_kw_std` |
| accept=0.00 peak reduction mean | -0.0 | % | `multiseed_results.json` | `coordinated/accept=0.00/peak_red_pct_mean` |
| accept=0.00 P95 reduction mean | 0.0 | % | `multiseed_results.json` | `coordinated/accept=0.00/p95_red_pct_mean` |
| accept=0.50 peak mean | 37.21 | kW | `multiseed_results.json` | `coordinated/accept=0.50/peak_kw_mean` |
| accept=0.50 peak sd | 3.183 | kW | `multiseed_results.json` | `coordinated/accept=0.50/peak_kw_std` |
| accept=0.50 P95 mean | 24.62 | kW | `multiseed_results.json` | `coordinated/accept=0.50/p95_kw_mean` |
| accept=0.50 P95 sd | 0.26 | kW | `multiseed_results.json` | `coordinated/accept=0.50/p95_kw_std` |
| accept=0.50 peak reduction mean | 7.83 | % | `multiseed_results.json` | `coordinated/accept=0.50/peak_red_pct_mean` |
| accept=0.50 P95 reduction mean | 8.12 | % | `multiseed_results.json` | `coordinated/accept=0.50/p95_red_pct_mean` |
| accept=0.85 peak mean | 27.94 | kW | `multiseed_results.json` | `coordinated/accept=0.85/peak_kw_mean` |
| accept=0.85 peak sd | 2.458 | kW | `multiseed_results.json` | `coordinated/accept=0.85/peak_kw_std` |
| accept=0.85 P95 mean | 19.21 | kW | `multiseed_results.json` | `coordinated/accept=0.85/p95_kw_mean` |
| accept=0.85 P95 sd | 0.491 | kW | `multiseed_results.json` | `coordinated/accept=0.85/p95_kw_std` |
| accept=0.85 peak reduction mean | 30.79 | % | `multiseed_results.json` | `coordinated/accept=0.85/peak_red_pct_mean` |
| accept=0.85 P95 reduction mean | 28.31 | % | `multiseed_results.json` | `coordinated/accept=0.85/p95_red_pct_mean` |
| accept=1.00 peak mean | 26.84 | kW | `multiseed_results.json` | `coordinated/accept=1.00/peak_kw_mean` |
| accept=1.00 peak sd | 0.0 | kW | `multiseed_results.json` | `coordinated/accept=1.00/peak_kw_std` |
| accept=1.00 P95 mean | 18.11 | kW | `multiseed_results.json` | `coordinated/accept=1.00/p95_kw_mean` |
| accept=1.00 P95 sd | 0.0 | kW | `multiseed_results.json` | `coordinated/accept=1.00/p95_kw_std` |
| accept=1.00 peak reduction mean | 33.52 | % | `multiseed_results.json` | `coordinated/accept=1.00/peak_red_pct_mean` |
| accept=1.00 P95 reduction mean | 32.42 | % | `multiseed_results.json` | `coordinated/accept=1.00/p95_red_pct_mean` |

## bootstrap

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| accept=0.00 peak_kw 95% CI | [40.37, 40.37] | kW | `stats_summary.json` | `accept_levels/accept=0.00/peak_kw/ci95` |
| accept=0.00 p95_kw 95% CI | [26.79, 26.79] | kW | `stats_summary.json` | `accept_levels/accept=0.00/p95_kw/ci95` |
| accept=0.00 peak_reduction_kw_boot_ci95 | [0.0, 0.0] | kW | `stats_summary.json` | `accept_levels/accept=0.00/peak_reduction_kw_boot_ci95` |
| accept=0.00 p95_reduction_kw_boot_ci95 | [0.0, 0.0] | kW | `stats_summary.json` | `accept_levels/accept=0.00/p95_reduction_kw_boot_ci95` |
| accept=0.50 peak_kw 95% CI | [34.81, 39.61] | kW | `stats_summary.json` | `accept_levels/accept=0.50/peak_kw/ci95` |
| accept=0.50 p95_kw 95% CI | [24.42, 24.82] | kW | `stats_summary.json` | `accept_levels/accept=0.50/p95_kw/ci95` |
| accept=0.50 peak_reduction_kw_boot_ci95 | [1.26, 5.14] | kW | `stats_summary.json` | `accept_levels/accept=0.50/peak_reduction_kw_boot_ci95` |
| accept=0.50 p95_reduction_kw_boot_ci95 | [2.02, 2.34] | kW | `stats_summary.json` | `accept_levels/accept=0.50/p95_reduction_kw_boot_ci95` |
| accept=0.85 peak_kw 95% CI | [26.09, 29.79] | kW | `stats_summary.json` | `accept_levels/accept=0.85/peak_kw/ci95` |
| accept=0.85 p95_kw 95% CI | [18.84, 19.58] | kW | `stats_summary.json` | `accept_levels/accept=0.85/p95_kw/ci95` |
| accept=0.85 peak_reduction_kw_boot_ci95 | [10.62, 13.7] | kW | `stats_summary.json` | `accept_levels/accept=0.85/peak_reduction_kw_boot_ci95` |
| accept=0.85 p95_reduction_kw_boot_ci95 | [7.25, 7.85] | kW | `stats_summary.json` | `accept_levels/accept=0.85/p95_reduction_kw_boot_ci95` |
| accept=1.00 peak_kw 95% CI | [26.84, 26.84] | kW | `stats_summary.json` | `accept_levels/accept=1.00/peak_kw/ci95` |
| accept=1.00 p95_kw 95% CI | [18.11, 18.11] | kW | `stats_summary.json` | `accept_levels/accept=1.00/p95_kw/ci95` |
| accept=1.00 peak_reduction_kw_boot_ci95 | [13.53, 13.53] | kW | `stats_summary.json` | `accept_levels/accept=1.00/peak_reduction_kw_boot_ci95` |
| accept=1.00 p95_reduction_kw_boot_ci95 | [8.68, 8.68] | kW | `stats_summary.json` | `accept_levels/accept=1.00/p95_reduction_kw_boot_ci95` |

## ladder

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| no_dr peak | 40.37 | kW | `ev_strategy_ladder.json` | `rows/no_dr/peak_kw` |
| no_dr P95 | 26.79 | kW | `ev_strategy_ladder.json` | `rows/no_dr/p95_kw` |
| random peak | 40.43 | kW | `ev_strategy_ladder.json` | `rows/random/peak_kw` |
| random P95 | 24.79 | kW | `ev_strategy_ladder.json` | `rows/random/p95_kw` |
| random peak reduction | -0.15 | % | `ev_strategy_ladder.json` | `rows/random/peak_red_pct` |
| random P95 reduction | 7.47 | % | `ev_strategy_ladder.json` | `rows/random/p95_red_pct` |
| edf peak | 32.41 | kW | `ev_strategy_ladder.json` | `rows/edf/peak_kw` |
| edf P95 | 18.65 | kW | `ev_strategy_ladder.json` | `rows/edf/p95_kw` |
| edf peak reduction | 19.72 | % | `ev_strategy_ladder.json` | `rows/edf/peak_red_pct` |
| edf P95 reduction | 30.38 | % | `ev_strategy_ladder.json` | `rows/edf/p95_red_pct` |
| stagger peak | 32.41 | kW | `ev_strategy_ladder.json` | `rows/stagger/peak_kw` |
| stagger P95 | 19.48 | kW | `ev_strategy_ladder.json` | `rows/stagger/p95_kw` |
| stagger peak reduction | 19.72 | % | `ev_strategy_ladder.json` | `rows/stagger/peak_red_pct` |
| stagger P95 reduction | 27.29 | % | `ev_strategy_ladder.json` | `rows/stagger/p95_red_pct` |
| mpc_bound peak | 24.45 | kW | `ev_strategy_ladder.json` | `rows/mpc_bound/peak_kw` |
| mpc_bound P95 | 24.45 | kW | `ev_strategy_ladder.json` | `rows/mpc_bound/p95_kw` |
| MPC ladder no_dr peak | 40.37 | kW | `mpc_ladder.json` | `no_dr/peak_kw` |
| MPC ladder rule_85 peak | 32.41 | kW | `mpc_ladder.json` | `rule_85/peak_kw` |
| MPC ladder rule_100 peak | 26.84 | kW | `mpc_ladder.json` | `rule_100/peak_kw` |
| MPC ladder mpc_bound peak | 24.45 | kW | `mpc_ladder.json` | `mpc_bound/peak_kw` |
| MPC LP energy conserved | True |  | `mpc_ladder.json` | `energy_conserved` |
| max attainable peak reduction | 15.92 | kW | `mpc_ladder.json` | `derived: no_dr/peak_kw - mpc_bound/peak_kw` |
| rule_85 share of the bound | 50.0 | % | `mpc_ladder.json` | `derived: (no_dr - rule_85)/(no_dr - mpc_bound)` |
| rule_100 share of the bound | 85.0 | % | `mpc_ladder.json` | `derived: (no_dr - rule_100)/(no_dr - mpc_bound)` |

## peak

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| baseline PAR | 3.9 |  | `peak_analysis.json` | `par/baseline/par` |
| independent PAR | 3.38 |  | `peak_analysis.json` | `par/independent/par` |
| coordinated PAR | 3.14 |  | `peak_analysis.json` | `par/coordinated/par` |
| PAR reduction (coordinated) | 19.68 | % | `peak_analysis.json` | `par/par_reduction_pct` |
| seeds in the low peak mode | 8 |  | `peak_analysis.json` | `ten_seed_peak/n_low_mode` |
| low-mode peak min | 26.33 | kW | `peak_analysis.json` | `ten_seed_peak/low_min_kw` |
| low-mode peak max | 26.84 | kW | `peak_analysis.json` | `ten_seed_peak/low_max_kw` |
| seeds in the high peak mode | 2 |  | `peak_analysis.json` | `ten_seed_peak/n_high_mode` |
| P95 sd as % of mean | 2.6 | % | `peak_analysis.json` | `ten_seed_peak/p95_sd_pct_of_mean` |
| peak sd as % of mean | 8.8 | % | `peak_analysis.json` | `ten_seed_peak/peak_sd_pct_of_mean` |
| high-mode peak 1 | 32.76 | kW | `peak_analysis.json` | `ten_seed_peak/high_values_kw/0` |
| high-mode peak 2 | 32.92 | kW | `peak_analysis.json` | `ten_seed_peak/high_values_kw/1` |
| low-mode peak reduction (min) | 33.5 | % | `peak_analysis.json` | `ten_seed_peak/low_reduction_pct_range/0` |
| low-mode peak reduction (max) | 34.8 | % | `peak_analysis.json` | `ten_seed_peak/low_reduction_pct_range/1` |
| high-mode peak reduction (min) | 18.5 | % | `peak_analysis.json` | `ten_seed_peak/high_reduction_pct_range/0` |
| high-mode peak reduction (max) | 18.9 | % | `peak_analysis.json` | `ten_seed_peak/high_reduction_pct_range/1` |
| full acceptance share of the bound | 85.0 | % | `peak_analysis.json` | `bound_share_pct/full_acceptance` |
| ten-seed mean share of the bound | 78.1 | % | `peak_analysis.json` | `bound_share_pct/ten_seed_mean` |
| reference run share of the bound | 50.0 | % | `peak_analysis.json` | `bound_share_pct/reference_run` |
| reference peak time | 2014-07-12 21:50 |  | `peak_analysis.json` | `reference_night/peak_time` |
| reference night proposals rejected | 2 |  | `peak_analysis.json` | `derived: count of reference_night/proposals_that_night with accepted=false` |

## factorial

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| scenarios | 360 | count | `ev_factorial.json` | `summary/n_scenarios` |
| acceptance rate | 0.85 |  | `ev_factorial.json` | `summary/accept_p` |
| random peak_red_pct mean | 6.97 | % | `ev_factorial.json` | `summary/strategies/random/peak_red_pct/mean` |
| random peak_red_pct sd | 10.8 | % | `ev_factorial.json` | `summary/strategies/random/peak_red_pct/sd` |
| random peak_red_pct min | -13.85 | % | `ev_factorial.json` | `summary/strategies/random/peak_red_pct/min` |
| random peak_red_pct max | 45.09 | % | `ev_factorial.json` | `summary/strategies/random/peak_red_pct/max` |
| random p95_red_pct mean | 10.34 | % | `ev_factorial.json` | `summary/strategies/random/p95_red_pct/mean` |
| random p95_red_pct sd | 8.52 | % | `ev_factorial.json` | `summary/strategies/random/p95_red_pct/sd` |
| random p95_red_pct min | -12.76 | % | `ev_factorial.json` | `summary/strategies/random/p95_red_pct/min` |
| random p95_red_pct max | 35.53 | % | `ev_factorial.json` | `summary/strategies/random/p95_red_pct/max` |
| random scenarios with worse peak | 46 | count | `ev_factorial.json` | `summary/strategies/random/n_peak_worse` |
| random scenarios with worse P95 | 23 | count | `ev_factorial.json` | `summary/strategies/random/n_p95_worse` |
| random scenarios with unchanged peak | 101 | count | `ev_factorial.json` | `summary/strategies/random/n_peak_unchanged` |
| random scenarios with improved peak | 213 | count | `ev_factorial.json` | `derived: n_scenarios - n_peak_unchanged - n_peak_worse (random)` |
| stagger peak_red_pct mean | 12.41 | % | `ev_factorial.json` | `summary/strategies/stagger/peak_red_pct/mean` |
| stagger peak_red_pct sd | 14.41 | % | `ev_factorial.json` | `summary/strategies/stagger/peak_red_pct/sd` |
| stagger peak_red_pct min | -14.79 | % | `ev_factorial.json` | `summary/strategies/stagger/peak_red_pct/min` |
| stagger peak_red_pct max | 44.51 | % | `ev_factorial.json` | `summary/strategies/stagger/peak_red_pct/max` |
| stagger p95_red_pct mean | 19.95 | % | `ev_factorial.json` | `summary/strategies/stagger/p95_red_pct/mean` |
| stagger p95_red_pct sd | 13.41 | % | `ev_factorial.json` | `summary/strategies/stagger/p95_red_pct/sd` |
| stagger p95_red_pct min | 0.69 | % | `ev_factorial.json` | `summary/strategies/stagger/p95_red_pct/min` |
| stagger p95_red_pct max | 49.8 | % | `ev_factorial.json` | `summary/strategies/stagger/p95_red_pct/max` |
| stagger scenarios with worse peak | 18 | count | `ev_factorial.json` | `summary/strategies/stagger/n_peak_worse` |
| stagger scenarios with worse P95 | 0 | count | `ev_factorial.json` | `summary/strategies/stagger/n_p95_worse` |
| stagger scenarios with unchanged peak | 112 | count | `ev_factorial.json` | `summary/strategies/stagger/n_peak_unchanged` |
| stagger scenarios with improved peak | 230 | count | `ev_factorial.json` | `derived: n_scenarios - n_peak_unchanged - n_peak_worse (stagger)` |
| edf peak_red_pct mean | 14.96 | % | `ev_factorial.json` | `summary/strategies/edf/peak_red_pct/mean` |
| edf peak_red_pct sd | 16.63 | % | `ev_factorial.json` | `summary/strategies/edf/peak_red_pct/sd` |
| edf peak_red_pct min | -22.38 | % | `ev_factorial.json` | `summary/strategies/edf/peak_red_pct/min` |
| edf peak_red_pct max | 52.42 | % | `ev_factorial.json` | `summary/strategies/edf/peak_red_pct/max` |
| edf p95_red_pct mean | 21.04 | % | `ev_factorial.json` | `summary/strategies/edf/p95_red_pct/mean` |
| edf p95_red_pct sd | 13.73 | % | `ev_factorial.json` | `summary/strategies/edf/p95_red_pct/sd` |
| edf p95_red_pct min | 0.69 | % | `ev_factorial.json` | `summary/strategies/edf/p95_red_pct/min` |
| edf p95_red_pct max | 51.57 | % | `ev_factorial.json` | `summary/strategies/edf/p95_red_pct/max` |
| edf scenarios with worse peak | 14 | count | `ev_factorial.json` | `summary/strategies/edf/n_peak_worse` |
| edf scenarios with worse P95 | 0 | count | `ev_factorial.json` | `summary/strategies/edf/n_p95_worse` |
| edf scenarios with unchanged peak | 107 | count | `ev_factorial.json` | `summary/strategies/edf/n_peak_unchanged` |
| edf scenarios with improved peak | 239 | count | `ev_factorial.json` | `derived: n_scenarios - n_peak_unchanged - n_peak_worse (edf)` |
| clustered arrivals, random, peak reduction | 8.15 | % | `ev_factorial.json` | `summary/by_arrival/clustered/random/peak_red_pct_mean` |
| clustered arrivals, random, P95 reduction | 13.75 | % | `ev_factorial.json` | `summary/by_arrival/clustered/random/p95_red_pct_mean` |
| clustered arrivals, random, peak worse | 24 | count | `ev_factorial.json` | `summary/by_arrival/clustered/random/n_peak_worse` |
| clustered arrivals, scenarios | 180 | count | `ev_factorial.json` | `summary/by_arrival/clustered/random/n` |
| clustered arrivals, stagger, peak reduction | 10.59 | % | `ev_factorial.json` | `summary/by_arrival/clustered/stagger/peak_red_pct_mean` |
| clustered arrivals, stagger, P95 reduction | 23.41 | % | `ev_factorial.json` | `summary/by_arrival/clustered/stagger/p95_red_pct_mean` |
| clustered arrivals, stagger, peak worse | 18 | count | `ev_factorial.json` | `summary/by_arrival/clustered/stagger/n_peak_worse` |
| clustered arrivals, scenarios | 180 | count | `ev_factorial.json` | `summary/by_arrival/clustered/stagger/n` |
| clustered arrivals, edf, peak reduction | 14.66 | % | `ev_factorial.json` | `summary/by_arrival/clustered/edf/peak_red_pct_mean` |
| clustered arrivals, edf, P95 reduction | 24.58 | % | `ev_factorial.json` | `summary/by_arrival/clustered/edf/p95_red_pct_mean` |
| clustered arrivals, edf, peak worse | 11 | count | `ev_factorial.json` | `summary/by_arrival/clustered/edf/n_peak_worse` |
| clustered arrivals, scenarios | 180 | count | `ev_factorial.json` | `summary/by_arrival/clustered/edf/n` |
| dispersed arrivals, random, peak reduction | 5.79 | % | `ev_factorial.json` | `summary/by_arrival/dispersed/random/peak_red_pct_mean` |
| dispersed arrivals, random, P95 reduction | 6.92 | % | `ev_factorial.json` | `summary/by_arrival/dispersed/random/p95_red_pct_mean` |
| dispersed arrivals, random, peak worse | 22 | count | `ev_factorial.json` | `summary/by_arrival/dispersed/random/n_peak_worse` |
| dispersed arrivals, scenarios | 180 | count | `ev_factorial.json` | `summary/by_arrival/dispersed/random/n` |
| dispersed arrivals, stagger, peak reduction | 14.23 | % | `ev_factorial.json` | `summary/by_arrival/dispersed/stagger/peak_red_pct_mean` |
| dispersed arrivals, stagger, P95 reduction | 16.49 | % | `ev_factorial.json` | `summary/by_arrival/dispersed/stagger/p95_red_pct_mean` |
| dispersed arrivals, stagger, peak worse | 0 | count | `ev_factorial.json` | `summary/by_arrival/dispersed/stagger/n_peak_worse` |
| dispersed arrivals, scenarios | 180 | count | `ev_factorial.json` | `summary/by_arrival/dispersed/stagger/n` |
| dispersed arrivals, edf, peak reduction | 15.27 | % | `ev_factorial.json` | `summary/by_arrival/dispersed/edf/peak_red_pct_mean` |
| dispersed arrivals, edf, P95 reduction | 17.5 | % | `ev_factorial.json` | `summary/by_arrival/dispersed/edf/p95_red_pct_mean` |
| dispersed arrivals, edf, peak worse | 3 | count | `ev_factorial.json` | `summary/by_arrival/dispersed/edf/n_peak_worse` |
| dispersed arrivals, scenarios | 180 | count | `ev_factorial.json` | `summary/by_arrival/dispersed/edf/n` |
| 3 EVs/night, random, P95 reduction | 5.78 | % | `ev_factorial.json` | `summary/by_factor/n_ev/3/random/p95_red_pct_mean` |
| 3 EVs/night, stagger, P95 reduction | 6.58 | % | `ev_factorial.json` | `summary/by_factor/n_ev/3/stagger/p95_red_pct_mean` |
| 3 EVs/night, edf, P95 reduction | 7.94 | % | `ev_factorial.json` | `summary/by_factor/n_ev/3/edf/p95_red_pct_mean` |
| 5 EVs/night, random, P95 reduction | 10.96 | % | `ev_factorial.json` | `summary/by_factor/n_ev/5/random/p95_red_pct_mean` |
| 5 EVs/night, stagger, P95 reduction | 19.03 | % | `ev_factorial.json` | `summary/by_factor/n_ev/5/stagger/p95_red_pct_mean` |
| 5 EVs/night, edf, P95 reduction | 20.02 | % | `ev_factorial.json` | `summary/by_factor/n_ev/5/edf/p95_red_pct_mean` |
| 10 EVs/night, random, P95 reduction | 14.27 | % | `ev_factorial.json` | `summary/by_factor/n_ev/10/random/p95_red_pct_mean` |
| 10 EVs/night, stagger, P95 reduction | 34.23 | % | `ev_factorial.json` | `summary/by_factor/n_ev/10/stagger/p95_red_pct_mean` |
| 10 EVs/night, edf, P95 reduction | 35.16 | % | `ev_factorial.json` | `summary/by_factor/n_ev/10/edf/p95_red_pct_mean` |

## llm

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| model | llama3.1:8b |  | `llm_eval_llama3.1_8b.json` | `summary/model` |
| events evaluated | 124 |  | `llm_eval_llama3.1_8b.json` | `summary/n_events` |
| call completion rate | 1.0 |  | `llm_eval_llama3.1_8b.json` | `summary/call_completion_rate` |
| transport failures | 0 |  | `llm_eval_llama3.1_8b.json` | `summary/n_transport_failures` |
| schema-valid rate | 1.0 |  | `llm_eval_llama3.1_8b.json` | `summary/schema_valid_rate_of_completed` |
| fully grounded message rate | 0.9435 |  | `llm_eval_llama3.1_8b.json` | `summary/msg_all_citations_grounded_rate` |
| ungrounded-number message rate | 0.1613 |  | `llm_eval_llama3.1_8b.json` | `summary/msg_with_ungrounded_number_rate` |
| unit-error message rate | 0.0 |  | `llm_eval_llama3.1_8b.json` | `summary/msg_with_unit_error_rate` |
| latency mean | 3.26 | s | `llm_eval_llama3.1_8b.json` | `summary/latency_s/mean` |
| latency p50 | 3.2 | s | `llm_eval_llama3.1_8b.json` | `summary/latency_s/p50` |
| latency p95 | 3.51 | s | `llm_eval_llama3.1_8b.json` | `summary/latency_s/p95` |
| Ollama version | 0.34.3 |  | `llm_eval_llama3.1_8b.json` | `server_env/ollama_version` |
| model digest | 46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e |  | `llm_eval_llama3.1_8b.json` | `server_env/model_digest` |
| model blob modified | 2026-06-27T15:32:24.4228566+08:00 |  | `llm_eval_llama3.1_8b.json` | `server_env/model_modified_at` |
| quantization | Q4_K_M |  | `llm_eval_llama3.1_8b.json` | `server_env/quantization` |
| parameter size | 8.0B |  | `llm_eval_llama3.1_8b.json` | `server_env/parameter_size` |
| repeats | 3 | runs | `llm_eval_llama3.1_8b.json` | `repeatability/n_repeats` |
| call_completion_rate over repeats | 1.0 [1.0, 1.0] |  | `llm_eval_llama3.1_8b.json` | `repeatability/per_metric/call_completion_rate` |
| schema_valid_rate_of_completed over repeats | 1.0 [1.0, 1.0] |  | `llm_eval_llama3.1_8b.json` | `repeatability/per_metric/schema_valid_rate_of_completed` |
| msg_all_citations_grounded_rate over repeats | 0.9435 [0.9435, 0.9435] |  | `llm_eval_llama3.1_8b.json` | `repeatability/per_metric/msg_all_citations_grounded_rate` |
| msg_with_ungrounded_number_rate over repeats | 0.1613 [0.1613, 0.1613] |  | `llm_eval_llama3.1_8b.json` | `repeatability/per_metric/msg_with_ungrounded_number_rate` |
| msg_with_unit_error_rate over repeats | 0.0 [0.0, 0.0] |  | `llm_eval_llama3.1_8b.json` | `repeatability/per_metric/msg_with_unit_error_rate` |

## llm contract

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| messages | 124 | count | `llm_contract.json` | `n_messages` |
| schema_ok pass rate | 1.0 |  | `llm_contract.json` | `per_check/schema_ok/rate` |
| schema_ok passed / applicable | 124/124 |  | `llm_contract.json` | `per_check/schema_ok` |
| numbers_grounded pass rate | 0.8952 |  | `llm_contract.json` | `per_check/numbers_grounded/rate` |
| numbers_grounded passed / applicable | 111/124 |  | `llm_contract.json` | `per_check/numbers_grounded` |
| citations_grounded pass rate | 0.9839 |  | `llm_contract.json` | `per_check/citations_grounded/rate` |
| citations_grounded passed / applicable | 122/124 |  | `llm_contract.json` | `per_check/citations_grounded` |
| citation_coverage pass rate | 0.879 |  | `llm_contract.json` | `per_check/citation_coverage/rate` |
| citation_coverage passed / applicable | 109/124 |  | `llm_contract.json` | `per_check/citation_coverage` |
| units_ok pass rate | 1.0 |  | `llm_contract.json` | `per_check/units_ok/rate` |
| units_ok passed / applicable | 124/124 |  | `llm_contract.json` | `per_check/units_ok` |
| currency_scale_ok pass rate | 0.8625 |  | `llm_contract.json` | `per_check/currency_scale_ok/rate` |
| currency_scale_ok passed / applicable | 69/80 |  | `llm_contract.json` | `per_check/currency_scale_ok` |
| time_explicit pass rate | 0.6591 |  | `llm_contract.json` | `per_check/time_explicit/rate` |
| time_explicit passed / applicable | 29/44 |  | `llm_contract.json` | `per_check/time_explicit` |
| wording_is_proposal pass rate | 1.0 |  | `llm_contract.json` | `per_check/wording_is_proposal/rate` |
| wording_is_proposal passed / applicable | 124/124 |  | `llm_contract.json` | `per_check/wording_is_proposal` |
| all checks passed | 92 | count | `llm_contract.json` | `all_checks_passed` |
| all checks pass rate | 0.7419 |  | `llm_contract.json` | `all_checks_pass_rate` |

## llm versions

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| messages compared | 124 | count | `llm_version_compare.json` | `messages_compared` |
| messages with different text | 104 | count | `llm_version_compare.json` | `messages_with_different_text` |
| messages with different text | 83.9 | % | `llm_version_compare.json` | `messages_with_different_text_pct` |
| baseline all-checks pass rate | 0.7258 |  | `llm_version_compare.json` | `baseline/all_checks_pass_rate` |
| baseline schema_ok | 1.0 |  | `llm_version_compare.json` | `baseline/per_check/schema_ok/rate` |
| baseline numbers_grounded | 0.9677 |  | `llm_version_compare.json` | `baseline/per_check/numbers_grounded/rate` |
| baseline citations_grounded | 0.9839 |  | `llm_version_compare.json` | `baseline/per_check/citations_grounded/rate` |
| baseline citation_coverage | 0.7419 |  | `llm_version_compare.json` | `baseline/per_check/citation_coverage/rate` |
| baseline units_ok | 1.0 |  | `llm_version_compare.json` | `baseline/per_check/units_ok/rate` |
| baseline currency_scale_ok | 0.9875 |  | `llm_version_compare.json` | `baseline/per_check/currency_scale_ok/rate` |
| baseline time_explicit | 0.4545 |  | `llm_version_compare.json` | `baseline/per_check/time_explicit/rate` |
| baseline wording_is_proposal | 1.0 |  | `llm_version_compare.json` | `baseline/per_check/wording_is_proposal/rate` |
| current all-checks pass rate | 0.7419 |  | `llm_version_compare.json` | `current/all_checks_pass_rate` |
| current schema_ok | 1.0 |  | `llm_version_compare.json` | `current/per_check/schema_ok/rate` |
| current numbers_grounded | 0.8952 |  | `llm_version_compare.json` | `current/per_check/numbers_grounded/rate` |
| current citations_grounded | 0.9839 |  | `llm_version_compare.json` | `current/per_check/citations_grounded/rate` |
| current citation_coverage | 0.879 |  | `llm_version_compare.json` | `current/per_check/citation_coverage/rate` |
| current units_ok | 1.0 |  | `llm_version_compare.json` | `current/per_check/units_ok/rate` |
| current currency_scale_ok | 0.8625 |  | `llm_version_compare.json` | `current/per_check/currency_scale_ok/rate` |
| current time_explicit | 0.6591 |  | `llm_version_compare.json` | `current/per_check/time_explicit/rate` |
| current wording_is_proposal | 1.0 |  | `llm_version_compare.json` | `current/per_check/wording_is_proposal/rate` |
| schema_ok change across builds | 0.0 |  | `llm_version_compare.json` | `rate_change_current_minus_baseline/schema_ok` |
| numbers_grounded change across builds | -0.0725 |  | `llm_version_compare.json` | `rate_change_current_minus_baseline/numbers_grounded` |
| citations_grounded change across builds | 0.0 |  | `llm_version_compare.json` | `rate_change_current_minus_baseline/citations_grounded` |
| citation_coverage change across builds | 0.1371 |  | `llm_version_compare.json` | `rate_change_current_minus_baseline/citation_coverage` |
| units_ok change across builds | 0.0 |  | `llm_version_compare.json` | `rate_change_current_minus_baseline/units_ok` |
| currency_scale_ok change across builds | -0.125 |  | `llm_version_compare.json` | `rate_change_current_minus_baseline/currency_scale_ok` |
| time_explicit change across builds | 0.2046 |  | `llm_version_compare.json` | `rate_change_current_minus_baseline/time_explicit` |
| wording_is_proposal change across builds | 0.0 |  | `llm_version_compare.json` | `rate_change_current_minus_baseline/wording_is_proposal` |
| current Ollama version | 0.34.3 |  | `llm_version_compare.json` | `current_server/ollama_version` |
| current model digest | 46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e |  | `llm_version_compare.json` | `current_server/model_digest` |

## forecast

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| CNN-LSTM mean MAE | 263.43 | W | `forecast_eval.json` | `mean_mae_lstm_w` |
| persistence mean MAE | 192.22 | W | `forecast_eval.json` | `mean_mae_persistence_w` |
| CNN-LSTM mean RMSE | 561.83 | W | `forecast_eval.json` | `mean_rmse_lstm_w` |
| persistence mean RMSE | 551.23 | W | `forecast_eval.json` | `mean_rmse_persistence_w` |
| houses where CNN-LSTM wins | 2 | of 15 | `forecast_eval.json` | `lstm_wins_n_houses` |
| paired samples per house | 1968 | 10-min | `forecast_eval.json` | `scored_steps_per_house` |
| lookback steps excluded | 48 | 10-min | `forecast_eval.json` | `lookback_steps_excluded` |

## sensitivity

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| cohort_15_primary households | 15 | count | `sensitivity_suite.json` | `cohorts/0/houses` |
| cohort_15_primary baseline peak | 40.37 | kW | `sensitivity_suite.json` | `cohorts/0/baseline/peak_kw` |
| cohort_15_primary coordinated peak | 32.41 | kW | `sensitivity_suite.json` | `cohorts/0/coordinated/peak_kw` |
| cohort_15_primary coordinated P95 | 18.65 | kW | `sensitivity_suite.json` | `cohorts/0/coordinated/p95_kw` |
| cohort_15_primary peak reduction | 19.72 | % | `sensitivity_suite.json` | `cohorts/0/peak_red_pct` |
| cohort_15_primary P95 reduction | 30.39 | % | `sensitivity_suite.json` | `cohorts/0/p95_red_pct` |
| cohort_16_with_H3 households | 16 | count | `sensitivity_suite.json` | `cohorts/1/houses` |
| cohort_16_with_H3 baseline peak | 40.5 | kW | `sensitivity_suite.json` | `cohorts/1/baseline/peak_kw` |
| cohort_16_with_H3 coordinated peak | 32.74 | kW | `sensitivity_suite.json` | `cohorts/1/coordinated/peak_kw` |
| cohort_16_with_H3 coordinated P95 | 18.95 | kW | `sensitivity_suite.json` | `cohorts/1/coordinated/p95_kw` |
| cohort_16_with_H3 peak reduction | 19.17 | % | `sensitivity_suite.json` | `cohorts/1/peak_red_pct` |
| cohort_16_with_H3 P95 reduction | 29.74 | % | `sensitivity_suite.json` | `cohorts/1/p95_red_pct` |
| cohort_17_all_nonsolar households | 17 | count | `sensitivity_suite.json` | `cohorts/2/houses` |
| cohort_17_all_nonsolar baseline peak | 40.68 | kW | `sensitivity_suite.json` | `cohorts/2/baseline/peak_kw` |
| cohort_17_all_nonsolar coordinated peak | 33.4 | kW | `sensitivity_suite.json` | `cohorts/2/coordinated/peak_kw` |
| cohort_17_all_nonsolar coordinated P95 | 19.01 | kW | `sensitivity_suite.json` | `cohorts/2/coordinated/p95_kw` |
| cohort_17_all_nonsolar peak reduction | 17.9 | % | `sensitivity_suite.json` | `cohorts/2/peak_red_pct` |
| cohort_17_all_nonsolar P95 reduction | 29.86 | % | `sensitivity_suite.json` | `cohorts/2/p95_red_pct` |
| train_seed_7 peak | 32.41 | kW | `sensitivity_suite.json` | `training_seeds/1/coordinated/peak_kw` |
| train_seed_7 P95 | 18.62 | kW | `sensitivity_suite.json` | `training_seeds/1/coordinated/p95_kw` |
| train_seed_123 peak | 32.41 | kW | `sensitivity_suite.json` | `training_seeds/2/coordinated/peak_kw` |
| train_seed_123 P95 | 18.54 | kW | `sensitivity_suite.json` | `training_seeds/2/coordinated/p95_kw` |

## season

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| summer (headline) window | 2014-04-30 to 2014-07-14 12:00 |  | `season/season_summary.json` | `0/window` |
| summer (headline) baseline peak | 40.37 | kW | `season/season_summary.json` | `0/baseline/peak_kw` |
| summer (headline) baseline P95 | 26.79 | kW | `season/season_summary.json` | `0/baseline/p95_kw` |
| summer (headline) coordinated peak | 32.41 | kW | `season/season_summary.json` | `0/coordinated/peak_kw` |
| summer (headline) coordinated P95 | 18.65 | kW | `season/season_summary.json` | `0/coordinated/p95_kw` |
| summer (headline) peak reduction | 19.7 | % | `season/season_summary.json` | `0/peak_red_pct` |
| summer (headline) P95 reduction | 30.4 | % | `season/season_summary.json` | `0/p95_red_pct` |
| summer (headline) (first 12 d) baseline peak | 40.37 | kW | `season/season_summary.json` | `0/first_12d/baseline/peak_kw` |
| summer (headline) (first 12 d) baseline P95 | 26.44 | kW | `season/season_summary.json` | `0/first_12d/baseline/p95_kw` |
| summer (headline) (first 12 d) coordinated peak | 26.33 | kW | `season/season_summary.json` | `0/first_12d/coordinated/peak_kw` |
| summer (headline) (first 12 d) coordinated P95 | 18.43 | kW | `season/season_summary.json` | `0/first_12d/coordinated/p95_kw` |
| summer (headline) (first 12 d) peak reduction | 34.8 | % | `season/season_summary.json` | `0/first_12d/peak_red_pct` |
| summer (headline) (first 12 d) P95 reduction | 30.3 | % | `season/season_summary.json` | `0/first_12d/p95_red_pct` |
| autumn window | 2014-08-02 to 2014-10-15 12:00 |  | `season/season_summary.json` | `1/window` |
| autumn test slots | 2016 | 10-min | `season/season_summary.json` | `1/test_slots` |
| autumn baseline peak | 41.08 | kW | `season/season_summary.json` | `1/baseline/peak_kw` |
| autumn baseline P95 | 31.55 | kW | `season/season_summary.json` | `1/baseline/p95_kw` |
| autumn coordinated peak | 30.74 | kW | `season/season_summary.json` | `1/coordinated/peak_kw` |
| autumn coordinated P95 | 19.89 | kW | `season/season_summary.json` | `1/coordinated/p95_kw` |
| autumn peak reduction | 25.2 | % | `season/season_summary.json` | `1/peak_red_pct` |
| autumn P95 reduction | 37.0 | % | `season/season_summary.json` | `1/p95_red_pct` |
| autumn (first 12 d) baseline peak | 40.12 | kW | `season/season_summary.json` | `1/first_12d/baseline/peak_kw` |
| autumn (first 12 d) baseline P95 | 29.43 | kW | `season/season_summary.json` | `1/first_12d/baseline/p95_kw` |
| autumn (first 12 d) coordinated peak | 30.74 | kW | `season/season_summary.json` | `1/first_12d/coordinated/peak_kw` |
| autumn (first 12 d) coordinated P95 | 19.55 | kW | `season/season_summary.json` | `1/first_12d/coordinated/p95_kw` |
| autumn (first 12 d) peak reduction | 23.4 | % | `season/season_summary.json` | `1/first_12d/peak_red_pct` |
| autumn (first 12 d) P95 reduction | 33.6 | % | `season/season_summary.json` | `1/first_12d/p95_red_pct` |
| winter window | 2014-12-16 to 2015-02-28 12:00 |  | `season/season_summary.json` | `2/window` |
| winter test slots | 2016 | 10-min | `season/season_summary.json` | `2/test_slots` |
| winter baseline peak | 41.67 | kW | `season/season_summary.json` | `2/baseline/peak_kw` |
| winter baseline P95 | 32.53 | kW | `season/season_summary.json` | `2/baseline/p95_kw` |
| winter coordinated peak | 30.3 | kW | `season/season_summary.json` | `2/coordinated/peak_kw` |
| winter coordinated P95 | 21.55 | kW | `season/season_summary.json` | `2/coordinated/p95_kw` |
| winter peak reduction | 27.3 | % | `season/season_summary.json` | `2/peak_red_pct` |
| winter P95 reduction | 33.8 | % | `season/season_summary.json` | `2/p95_red_pct` |
| winter (first 12 d) baseline peak | 41.67 | kW | `season/season_summary.json` | `2/first_12d/baseline/peak_kw` |
| winter (first 12 d) baseline P95 | 32.64 | kW | `season/season_summary.json` | `2/first_12d/baseline/p95_kw` |
| winter (first 12 d) coordinated peak | 30.3 | kW | `season/season_summary.json` | `2/first_12d/coordinated/peak_kw` |
| winter (first 12 d) coordinated P95 | 21.81 | kW | `season/season_summary.json` | `2/first_12d/coordinated/p95_kw` |
| winter (first 12 d) peak reduction | 27.3 | % | `season/season_summary.json` | `2/first_12d/peak_red_pct` |
| winter (first 12 d) P95 reduction | 33.2 | % | `season/season_summary.json` | `2/first_12d/p95_red_pct` |
| spring window | 2015-02-13 to 2015-04-26 12:00 |  | `season/season_summary.json` | `3/window` |
| spring test slots | 1728 | 10-min | `season/season_summary.json` | `3/test_slots` |
| spring baseline peak | 40.79 | kW | `season/season_summary.json` | `3/baseline/peak_kw` |
| spring baseline P95 | 26.97 | kW | `season/season_summary.json` | `3/baseline/p95_kw` |
| spring coordinated peak | 30.63 | kW | `season/season_summary.json` | `3/coordinated/peak_kw` |
| spring coordinated P95 | 19.29 | kW | `season/season_summary.json` | `3/coordinated/p95_kw` |
| spring peak reduction | 24.9 | % | `season/season_summary.json` | `3/peak_red_pct` |
| spring P95 reduction | 28.5 | % | `season/season_summary.json` | `3/p95_red_pct` |
| spring (first 12 d) baseline peak | 40.79 | kW | `season/season_summary.json` | `3/first_12d/baseline/peak_kw` |
| spring (first 12 d) baseline P95 | 26.97 | kW | `season/season_summary.json` | `3/first_12d/baseline/p95_kw` |
| spring (first 12 d) coordinated peak | 30.63 | kW | `season/season_summary.json` | `3/first_12d/coordinated/peak_kw` |
| spring (first 12 d) coordinated P95 | 19.29 | kW | `season/season_summary.json` | `3/first_12d/coordinated/p95_kw` |
| spring (first 12 d) peak reduction | 24.9 | % | `season/season_summary.json` | `3/first_12d/peak_red_pct` |
| spring (first 12 d) P95 reduction | 28.5 | % | `season/season_summary.json` | `3/first_12d/p95_red_pct` |

## ablation

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| forecast: forecast=lstm peak | 32.41 | kW | `ablation_results.json` | `forecast/0/peak_kw` |
| forecast: forecast=lstm P95 reduction | 30.39 | % | `ablation_results.json` | `forecast/0/p95_reduction` |
| forecast: forecast=lstm decisions | 124 | count | `ablation_results.json` | `forecast/0/total_recs` |
| forecast: forecast=persistence peak | 32.41 | kW | `ablation_results.json` | `forecast/1/peak_kw` |
| forecast: forecast=persistence P95 reduction | 31.05 | % | `ablation_results.json` | `forecast/1/p95_reduction` |
| forecast: forecast=persistence decisions | 62 | count | `ablation_results.json` | `forecast/1/total_recs` |
| accept_rate: accept=0.00 peak | 40.37 | kW | `ablation_results.json` | `accept_rate/0/peak_kw` |
| accept_rate: accept=0.00 P95 reduction | 0.0 | % | `ablation_results.json` | `accept_rate/0/p95_reduction` |
| accept_rate: accept=0.00 decisions | 127 | count | `ablation_results.json` | `accept_rate/0/total_recs` |
| accept_rate: accept=0.50 peak | 33.08 | kW | `ablation_results.json` | `accept_rate/1/peak_kw` |
| accept_rate: accept=0.50 P95 reduction | 7.67 | % | `ablation_results.json` | `accept_rate/1/p95_reduction` |
| accept_rate: accept=0.50 decisions | 126 | count | `ablation_results.json` | `accept_rate/1/total_recs` |
| accept_rate: accept=0.85 peak | 32.41 | kW | `ablation_results.json` | `accept_rate/2/peak_kw` |
| accept_rate: accept=0.85 P95 reduction | 30.39 | % | `ablation_results.json` | `accept_rate/2/p95_reduction` |
| accept_rate: accept=0.85 decisions | 124 | count | `ablation_results.json` | `accept_rate/2/total_recs` |
| accept_rate: accept=1.00 peak | 26.84 | kW | `ablation_results.json` | `accept_rate/3/peak_kw` |
| accept_rate: accept=1.00 P95 reduction | 32.42 | % | `ablation_results.json` | `accept_rate/3/p95_reduction` |
| accept_rate: accept=1.00 decisions | 123 | count | `ablation_results.json` | `accept_rate/3/total_recs` |
| closed_loop: closedloop=off peak | 32.41 | kW | `ablation_results.json` | `closed_loop/0/peak_kw` |
| closed_loop: closedloop=off P95 reduction | 30.39 | % | `ablation_results.json` | `closed_loop/0/p95_reduction` |
| closed_loop: closedloop=off decisions | 124 | count | `ablation_results.json` | `closed_loop/0/total_recs` |
| closed_loop: closedloop=on peak | 32.41 | kW | `ablation_results.json` | `closed_loop/1/peak_kw` |
| closed_loop: closedloop=on P95 reduction | 30.39 | % | `ablation_results.json` | `closed_loop/1/p95_reduction` |
| closed_loop: closedloop=on decisions | 124 | count | `ablation_results.json` | `closed_loop/1/total_recs` |
| closed_loop: closedloop=stress peak | 32.41 | kW | `ablation_results.json` | `closed_loop/2/peak_kw` |
| closed_loop: closedloop=stress P95 reduction | 30.64 | % | `ablation_results.json` | `closed_loop/2/p95_reduction` |
| closed_loop: closedloop=stress decisions | 44 | count | `ablation_results.json` | `closed_loop/2/total_recs` |

## fairness

| Quantity | Value | Unit | Source file | JSON path |
|---|---|---|---|---|
| budget=inf Jain (total) | 0.6351 |  | `fairness_sweep.json` | `0/fairness` |
| budget=inf Jain (appliance) | 0.5116 |  | `fairness_sweep.json` | `0/fairness_appliance` |
| budget=inf P95 reduction | 30.39 | % | `fairness_sweep.json` | `0/p95_reduction` |
| budget=inf decisions skipped | 0 | count | `fairness_sweep.json` | `0/n_skipped_by_fairness` |
| budget=6 Jain (total) | 0.6351 |  | `fairness_sweep.json` | `1/fairness` |
| budget=6 Jain (appliance) | 0.5116 |  | `fairness_sweep.json` | `1/fairness_appliance` |
| budget=6 P95 reduction | 30.39 | % | `fairness_sweep.json` | `1/p95_reduction` |
| budget=6 decisions skipped | 0 | count | `fairness_sweep.json` | `1/n_skipped_by_fairness` |
| budget=4 Jain (total) | 0.6351 |  | `fairness_sweep.json` | `2/fairness` |
| budget=4 Jain (appliance) | 0.5116 |  | `fairness_sweep.json` | `2/fairness_appliance` |
| budget=4 P95 reduction | 30.39 | % | `fairness_sweep.json` | `2/p95_reduction` |
| budget=4 decisions skipped | 0 | count | `fairness_sweep.json` | `2/n_skipped_by_fairness` |
| budget=2 Jain (total) | 0.6392 |  | `fairness_sweep.json` | `3/fairness` |
| budget=2 Jain (appliance) | 0.5291 |  | `fairness_sweep.json` | `3/fairness_appliance` |
| budget=2 P95 reduction | 30.39 | % | `fairness_sweep.json` | `3/p95_reduction` |
| budget=2 decisions skipped | 3 | count | `fairness_sweep.json` | `3/n_skipped_by_fairness` |
| budget=1 Jain (total) | 0.6473 |  | `fairness_sweep.json` | `4/fairness` |
| budget=1 Jain (appliance) | 0.5634 |  | `fairness_sweep.json` | `4/fairness_appliance` |
| budget=1 P95 reduction | 29.98 | % | `fairness_sweep.json` | `4/p95_reduction` |
| budget=1 decisions skipped | 22 | count | `fairness_sweep.json` | `4/n_skipped_by_fairness` |
