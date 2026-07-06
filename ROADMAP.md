# Roadmap — Smart-Home DR: single-household → multi-household → coordination

Strategy: **build & validate ONE single-household agent → add an LLM advisory
layer → scale to multi-household with coordination → learned coordination (MARL)
+ federated learning.** Build the single unit before the system.

Two code trees: **`conference/`** (single-household, published) and
**`multi_household/`** (the journal system, current focus).

---

## Phase 1 — Single-household forecasting ✅ DONE (conference)

Reproduce + improve Durrani et al. (2025) on UCI Appliances.
- [x] 8 forecasters (LR/RF/SVR/kNN/LSTM/Persistence/Seasonal-Naive/ETS) × 7 DR strategies
- [x] Honest finding: paper's R²=0.94 is a granularity artefact; 10-min R² ~0.6
- [x] CNN-LSTM v2 ensemble; E1–E6 experiment suite
- **Code:** `conference/src/{data,forecasting,dr,evaluation,viz}`

## Phase 2 — Single-household agent ✅ DONE (conference)

`forecast → state → decision → DR action → reward` on one home (simulated price/physics).
- [x] Gym-style env, rule-based controller, **MPC baseline** (LP receding-horizon, perfect-foresight upper bound)
- [x] v1 LLM advisory (schema-constrained + post-validator, anti-hallucination)
- [ ] RL agent (`conference/src/agent/rl_agent.py`) — scaffold only, deferred
- **Reused later:** `conference/src/agent/{mpc,rl_agent,env}.py` become multi-household controller baselines.

---

## Phase 3 — Multi-household coordination ✅ DONE (journal, `multi_household/`)

- [x] REFIT 16-house data: deglitch + clean window + **common-grid time alignment**
- [x] Per-house CNN-LSTM (local-only, train-only scaler — no leakage)
- [x] Aggregator: dynamic ToU price + peak detection + **hold-release** (anti-rebound)
- [x] Appliance-aware controller: rising-edge defer, cooldown, comfort cap, off-peak drain
- [x] **LLM advisory v2**: personalized, **Llama 3.1 (local Ollama)**, fact-citation + unit validation
- [x] **Closed-loop learning**: accept/reject/modify → agent suppresses rejected patterns
- [x] **EV advisory coordinator**: stagger the 5 EVs across the overnight trough,
      accept-gated (a recommendation, not automatic) → acceptance drives the peak
- [x] 54 unit tests; seeded ablations on clean data
- **Validated (85% accept):** coordinated peak **−19%**, P95 **−29%**, energy conserved,
  valley-filled (load-leveled). Full acceptance → **−29% peak**.

## Phase 4 — Stronger results + learned coordination ◀ NEXT

- [ ] **Closed-loop learning** as a focused report (R13) — accept-rate 99.6→85→45%
- [ ] **Controller baselines**: wire `conference/src/agent/{mpc,rl_agent}` to multi-house → table `No-DR | Rule(ours) | MPC | RL`
- [ ] **Federated learning** (Local vs FedAvg vs Centralized)
- [ ] **Seq2Seq 24 h forecast** (replace recursive, stop error compounding)
- [ ] **MARL** — replace hand-written rules with learned multi-agent coordination (Q1 novelty)

---

## Status snapshot

| Layer | Status |
|---|---|
| Single-household forecast + agent + v1 advisory | ✅ done (conference) |
| Multi-household data + per-house forecasting | ✅ done |
| Aggregator coordination + appliance control | ✅ done (−19% peak, −29% P95) |
| LLM advisory v2 + closed-loop learning | ✅ done (Llama 3.1, local) |
| EV advisory coordinator (accept-gated stagger) | ✅ done (drives −29% at full accept) |
| Controller baselines (MPC/RL multi-house) | ☐ next |
| FL · Seq2Seq · MARL | ☐ future |

## Recent (2026-07-03)

- **EV advisory coordinator enabled**: the (previously disabled) EV stagger is now
  wired in as an **accept-gated recommendation** → P95 −11.8% → **−29%**, headline
  peak unchanged (−19% at 85% accept), full acceptance → −29% peak. Energy conserved.
  Acceptance now cleanly drives the peak (0%→0%), strengthening the human-in-the-loop story.
- **Project cleanup**: removed all external-comparison side-work (an experimental
  shadow-price/oracle coordinator that a collaborator had contributed) so
  `multi_household/` is a single clean trunk — our own system only. 54 tests pass (2 new for the EV advisory).
- Repo restructured earlier: conference work → `conference/`, top level clean.
- LLM swapped qwen3 → **local Llama 3.1** (non-Chinese, offline, privacy).
- Ablations re-run on clean data with fixed seed.
- Reported through **R12 (0703)**: coordination deep-dive + LLM advisory v2.
