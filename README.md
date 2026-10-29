# Self-Healing Machines (HealLoop) — closed-loop predictive maintenance on NASA C-MAPSS (FD001–FD004)

Team Teliport · Case Study 1 (Tata ELXSI) — *ingest → detect degradation → estimate calibrated RUL → triage → auto-create work order; engineers handle exceptions.*

This repository turns the Round-1 design deck into a **working, end-to-end proof of concept** on all four C-MAPSS subsets.
No LLM and no agent framework anywhere: the agent is plain Python rules. Open-source stack only
(Python · Pandas/NumPy · scikit-learn · LightGBM · MLflow · Streamlit · Docker).

> **Design principle (from the deck):** *late RUL overestimation is more costly than early alerts; prevent false alarms and never let
> predictive software command an unsafe shutdown.* P1 tickets ask a human to take the safe-stop decision — the software never does.

## Quick start

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m selfheal run-all          # download -> train -> retrain demo -> replay -> fault replay -> report   (~45 min on a 6-core laptop;
                                    #   --subsets FD001 --skip-retrain is a ~4 min smoke run; most time is the cycle-by-cycle replays)
python -m selfheal dashboard        # http://localhost:8501
python -m selfheal mlflow-ui        # http://localhost:5000   (experiments, registry, rollback lineage)
python -m pytest                    # 46 tests (incl. 10 headless dashboard tests)
```

Docker (one command): `docker compose up --build` runs the pipeline, then serves the dashboard (:8501) and MLflow UI (:5000).
*The Docker files were written but could not be executed on the machine this was built on (no Docker installed) — see "Honest notes".*

| Command | What it does |
|---|---|
| `python -m selfheal download` | fetch the official NASA files into `data/raw/` (PHM-Society S3 mirror of the NASA repository) |
| `python -m selfheal train [--subset FD002] [--force]` | train, track in MLflow, register, promote the first champion |
| `python -m selfheal replay --scenario clean faults` | stream the **test** fleet one cycle at a time through the agent; writes tickets, scores, drift/quality reports |
| `python -m selfheal retrain` | released completed histories → challenger → grouped gate → promote or keep champion |
| `python -m selfheal rollback --subset FD001` | point the champion alias back at the previous version |
| `python -m selfheal status` / `report` | champions + registry versions / regenerate `outputs/RESULTS.md` |

## Architecture (deck → code)

```
 historical TRAIN + per-cycle replay
   1 INGEST ─► 2 CLEAN ─► 3 FEATURES ─► 4 MODELS ─► 5 TRIAGE AGENT ─► 6 WORK ORDER ─► human ACK / close
  data.py     clean.py   features.py   anomaly.py    triage.py        tickets.py       dashboard
                                        rul.py                                              │
   └────────────────────── MLOps feedback loop: mlops.py · monitor.py · pipeline.py ◄───────┘
```

| Deck element | Implementation |
|---|---|
| Parse 26 columns, preserve order, schema / range / gap / duplicate checks; reject duplicate or out-of-order cycles | `data.py` — `validate_frame` (batch, hard errors) and `CycleValidator` (streaming, rejects at ingest) |
| Causal short-gap forward-fill + flags, train-only median for initial gaps, long gaps → human review | `clean.py` — limited causal fill (`max_ffill_gap=3`), explicit missing-row insertion, `long_gap` flag held for 5 cycles |
| Drop near-constant sensors using TRAIN variance | pooled *within-regime* train std (a global variance would be dominated by the 6 operating regimes) → FD001 drops s1,s5,s10,s16,s18,s19 |
| Regime-aware scaling; far-from-train regimes → review | FD001/003: 1 regime; FD002/004: k-means(6) on train settings, per-regime median/std; distance beyond `q99.95 × 1.5` → `unseen_regime` |
| Causal 5/10/20/30 mean, SD, slope, extrema, EWMA, delta; start-up fallback | `features.py` — a pure function of the last 30 cleaned rows (NaN-padded at start-up) → batch == streaming, no leakage |
| Group train/validation by engine id; fit transforms on train; reserve test labels | `train.split_engines` (fit/calibration/validation engines are disjoint); cleaner, IF, PCA and models are fitted on fit engines only |
| Isolation Forest + PCA-SPE on first 20 cycles of TRAIN engines; healthy-validation percentile A; 3-of-5 persistence; tuned on early-life alerts / lead | `anomaly.py` — fused z-scores mapped to the percentile of held-out calibration-engine healthy cycles; `(A, k-of-n)` tuned on validation engines |
| LightGBM quantile P10/P50/P90, near-failure re-weighting, explainable baseline | `rul.py` — weight candidates `[1, 3]` chosen on validation; TreeSHAP (`pred_contrib`) for ticket evidence |
| Nominal 80 % interval calibrated by split conformal on held-out engines; report coverage **and** width | `rul.cqr_qhat` (CQR, Romano et al.) on calibration engines |
| RMSE + NASA score on locked test endpoints, FD001–FD004 separately | `train.evaluate_test`; baselines: constant, age-based, ridge on the same features |
| Triage rules, first match wins; one open incident per engine; de-escalate after 5 stable cycles | `triage.py` (+ vectorised twin used for tuning, tested equal to the scalar rules) |
| Idempotent P1–P3 JSON/CSV ticket with due cycle, evidence, model/rule ids; ACK/close | `tickets.py` — SQLite with a partial unique index (one open ticket per engine), deterministic ids, JSON + CSV export |
| MLflow: split ids, scalers, thresholds, metrics, versions, rollback | `mlops.py` — sqlite backend, registry aliases `champion` / `previous` |
| Unlabelled cycles → drift only; labelled trajectories → retrain + grouped challenger gate | `monitor.py` (PSI, quality counters) · `pipeline.retrain_demo` |
| Streamlit: data quality/drift, RUL bands, severity, live tickets, model version | `app/dashboard.py` (+ a *Live simulation* page with fault injection) |

### Triage rules (`triage.py`, first match wins)

| # | Rule | When | Action |
|---|---|---|---|
| 1 | REVIEW | bad / stale / gapped data, out-of-range sensors, unseen regime, model failure | exception ticket, engineer checks data |
| 2 | CRITICAL | `P50 ≤ crit` **or** (`P10 ≤ crit` **and** sustained anomaly) | **P1**, supervisor alert, safe-stop decision by a human (immediate — no persistence needed) |
| 3 | REVIEW | `P90−P10 > W*` **and** the band spans risk tiers; **or** sustained anomaly while the RUL model says healthy | review ticket + evidence |
| 4 | HIGH | `P10 ≤ high` | **P2**, due within 10 cycles |
| 5 | MEDIUM | `P10 ≤ med` **or** sustained anomaly | **P3**, next scheduled service |
| 6 | MONITOR | otherwise | dashboard only |

The cut-offs (default 15/40/80), `W*` and the anomaly alert rule are **validation-tuned policy candidates** (per subset, see `RESULTS.md`), not measured results.
Guardrails: one open incident per engine (escalate / update, never duplicate); bad data can annotate but never de-escalate a live P1/P2;
a human override is logged and blocks automatic de-escalation; closed tickets have a re-open cool-down; every ticket carries the rule id, policy version and model version.

## Results (final champions, run on this machine)

Official NASA test endpoints (one RUL per engine; evaluated once per trained model, never used for tuning). *RMSE vs capped* compares with the RUL clipped at
the 125-cycle modelling cap, which is the quantity the models are trained to predict (see "Honest notes").

| Subset | RMSE (official RUL) | RMSE vs capped | NASA score | RMSE RUL ≤ 30 | Ridge RMSE (baseline) | Age-based RMSE | Constant RMSE | 80 % interval coverage / width |
|---|---|---|---|---|---|---|---|---|
| FD001 | **13.61** | 12.70 | **314** | 6.18 | 16.3 | 37.3 | 52.1 | 0.83 / 35 |
| FD002 | **25.36** | 13.35 | **6848** | 3.95 | 27.9 | 41.0 | 57.7 | 0.90 / 35 |
| FD003 | **15.36** | 13.87 | **397** | 4.09 | 17.9 | 45.5 | 64.7 | 0.79 / 34 |
| FD004 | **26.40** | 14.72 | **4894** | 4.98 | 31.1 | 59.1 | 64.0 | 0.83 / 31 |

The agent on the **official test fleet**, streamed one cycle at a time (clean stream, final champion):

| Subset | engines | tickets | engines with true RUL ≤ 30 | … flagged P1/P2 at some point | … P1/P2 at last cycle | engines with true RUL > 80 | … of which P1/P2 at last cycle | drift | data quality |
|---|---|---|---|---|---|---|---|---|---|
| FD001 | 100 | 62 | 25 | **25** | 25 | 55 | 1 | ok | ok |
| FD002 | 259 | 171 | 61 | **61** | 61 | 129 | 3 | ok | ok |
| FD003 | 100 | 65 | 20 | **20** | 20 | 47 | 0 | warn | ok |
| FD004 | 248 | 152 | 53 | **53** | 53 | 135 | 2 | ok | ok |

No ticket is created in the first 10 cycles of any engine (start-up guard). On the validation engines (full run-to-failure replay) the first P1/P2 ticket fires a median of 38–56 cycles
before failure with RUL ≤ 30 window recall ≥ 0.998; the tuned policy per subset is in `outputs/RESULTS.md`.

**Retraining demo** — reserved *completed* run-to-failure histories are released, the agent first works them to failure and a simulated engineer closes each ticket with a
ground-truth-verified outcome, then the challenger is trained and gated on held-out engines. All four challengers passed the gate and were promoted (v1 → v2):

| Subset | released engines | held-out NASA score (mean) | held-out RMSE | official-test RMSE (informational) | official-test NASA (informational) | verified P1/P2 precision | median lead of first P1/P2 |
|---|---|---|---|---|---|---|---|
| FD001 | 25 | 7.48 → 7.08 | 17.69 → 16.85 | 15.53 → 13.61 | 414 → 314 | 0.92 (23 confirmed, 2 false alarms) | 49 cycles |
| FD002 | 65 | 8.10 → 6.83 | 18.46 → 17.64 | 26.27 → 25.36 | 8628 → 6848 | 0.91 (59 / 6) | 50 |
| FD003 | 25 | 4.82 → 4.00 | 16.35 → 15.49 | 15.96 → 15.36 | 413 → 397 | 1.00 (25 / 0) | 37 |
| FD004 | 62 | 7.18 → 6.17 | 18.02 → 17.27 | 26.41 → 26.40 | 4620 → 4894 | 0.85 (53 / 9) | 50 |

The gate improvements carried over to the untouched test set for FD001–FD003; on FD004 test RMSE is flat and the test NASA score is 6 % *worse*, i.e. the held-out gate is a
reasonable but imperfect proxy. Fault-injection replay (data gap, frozen feed, unseen regime, duplicate row, sensor offset on half the fleet): every injected fault becomes a REVIEW
exception ticket (e.g. FD001: gap → cycle 84, frozen feed → 51, unseen regime → 112, duplicate → 33) and the drift monitor alerts on the shifted sensor (PSI ≈ 1.0).

## MLOps loop

| Piece | How |
|---|---|
| **Tracking** | every training run logs params (full merged config), metrics, the grouped engine-id split, scalers / regime centres / OOD threshold / policy cut-offs / conformal `q̂` (`thresholds_and_scalers.json`), tuning tables and the model bundle to MLflow (sqlite backend in `mlflow/`) |
| **Versions & rollback** | one registered model per subset (`shm-rul-FD00x`); alias `champion` = serving model, `previous` = rollback target; `python -m selfheal rollback` swaps them (also mirrored in `artifacts/<subset>/champion.json` so the dashboard works without an MLflow server) |
| **Reproducibility** | fixed seeds, per-subset configs (`configs/FD00x.yaml` over `base.yaml`), identical `Cleaner` / `window_features` code for training and inference (asserted in `tests/test_pipeline.py::test_streaming_matches_batch`) |
| **Retraining flow** | 25 % of the TRAIN engines are *reserved* and withheld from the first model. When retraining runs they are released as completed, labelled run-to-failure histories: the agent first works them to failure and a simulated engineer closes each ticket with an outcome verified against ground truth (feedback precision / lead time), then a **challenger** is trained on fit + released engines. Calibration and validation engines are identical for champion and challenger. The official test set is never used for training or for the decision |
| **Gate (promote only with a better risk/error trade-off)** | challenger must cut the endpoint-like NASA score by ≥ 1 %, not worsen RMSE by > 2 % or the RUL ≤ 30 window RMSE by > 5 %, and keep 80 % coverage within 5 pts. Otherwise it is registered but **not** promoted (`challenger_rejected` in `history.json`) |
| **Unlabelled cycles** | feed **drift monitoring only** (`monitor.py`), never supervised retraining |

### Monitoring (dashboard → *Data quality & drift*)
* **Drift** — PSI per kept sensor between the incoming fleet's *early-life* cycles and the train early-life reference (warn ≥ 0.10, alert ≥ 0.25). Comparing like life-stage with like stops normal ageing from looking like drift.
* **Data quality** — counters for rejected rows (duplicate / out-of-order), long gaps, frozen feeds, out-of-range sensors, unseen regimes, imputed values.
* `--scenario faults` injects a long data gap, a frozen feed, an unseen operating regime, a duplicate row and a sensor offset on half the fleet, to show the agent handing each to a human (REVIEW exception tickets) and the drift monitor raising an alert.

### Dashboard
`python -m selfheal dashboard` — *Fleet overview* (severity map, rewindable fleet clock) · *Engine detail* (P10–P90 band vs ground truth, anomaly percentile, sensors, ticket audit trail) · *Work orders* (filter, **ACK / close / override** — the human in the loop, CSV download) · *Data quality & drift* · *Model & MLOps* (metrics vs baselines, registry, retrain gate, rollback, feature gain) · *Live simulation* (steps the real scorer + agent through the test fleet and lets you inject faults).

## Honest notes (what to say in the review)

* **Test discipline.** Transforms, anomaly detector, models, conformal calibration and policy cut-offs are fitted/tuned on TRAIN engines only (grouped by engine id); the official test endpoints are evaluated once per trained model. While *debugging* I did look at test numbers (e.g. to find the degenerate-booster and cap-sensitivity issues below) but no hyper-parameter, cap, threshold or window was chosen from them. One design change — endpoint-like conformal calibration — was motivated by seeing under-coverage on test endpoints, so test coverage is not a fully blind confirmation of that choice (validation coverage moved the same way).
* **RUL cap = 125** is a modelling choice from the deck. Models trained on a capped label cannot predict above 125, so for FD002/FD004 (22 % / 27 % of test engines have true RUL > 125, up to 195) the RMSE against the *raw* official RUL is inflated; we therefore report RMSE against the capped truth as well (`rmse_vs_capped_truth` in the dashboard / MLflow). A cap of 150 reduces the raw FD002/FD004 RMSE in a diagnostic run, but validation could not select it reliably (the answer flipped with the evaluation window), so 125 was kept for all four subsets.
* **Three fixes worth knowing about** (all covered by comments in the code): (1) the P90 booster never split when fitted on the *capped* label (start value = cap, identical gradients) — P10/P90 are now fitted on the uncapped RUL and clipped, which is the same quantile; (2) the median booster has the same failure when ≥ 50 % of labels sit on the cap (FD003) — fixed with `boost_from_average=False`; (3) conformal calibration on *all* rows is dominated by trivially covered healthy rows, so calibration/validation use "endpoint-like" rows (≥ 31 cycles of history, RUL ≤ 150, every engine weighted equally).
* **Deliberate deviation from the deck table:** a rule-3 REVIEW ticket inherits the priority that rules 4/5 *would* have assigned (so an uncertainty review can never silently demote a HIGH-eligible engine). Rule 3's "model conflict" is implemented as *sustained anomaly while the RUL model says healthy* (P10 ≥ 110), the only conflict the design can evaluate without the optional challenger. Start-up rows (< 10 cycles of history) skip the anomaly and uncertainty rules but never CRITICAL.
* **Not implemented:** the optional M2 1D-CNN challenger (needs PyTorch/TensorFlow; the deck says adopt it only if held-out performance justified the complexity). SHAP-style evidence is TreeSHAP on the LightGBM P50 model — indicative, **not** proof of root cause. Fault type is not labelled per engine in C-MAPSS, so no diagnosis is claimed.
* **Small-sample caution.** FD001/FD003 have 20 validation engines; the tuned cut-offs, `W*` and anomaly rule are *policy candidates* and will move with a different split. Ticket-level numbers come from simulated engines.
* **Docker** files (`Dockerfile`, `docker-compose.yml`) were written but not executed (Docker was not installed on the build machine). Everything else was run end-to-end on Windows 11 / Python 3.13.

## Project layout

```
selfheal/            data.py clean.py features.py anomaly.py rul.py metrics.py   (stages 1-4)
                     triage.py tickets.py tuning.py                              (stages 5-6, policy tuning)
                     scorer.py replay.py bundle.py train.py                      (streaming loop, training)
                     mlops.py pipeline.py monitor.py report.py download.py cli.py
configs/             base.yaml + FD001..FD004.yaml (per-subset overrides)
app/dashboard.py     Streamlit dashboard
tests/               46 tests: ingest, causality, streaming==batch, rules, agent invariants, MLflow registry, dashboard pages
data/raw/            NASA C-MAPSS (downloaded)         artifacts/  versioned model bundles + champion pointers
outputs/             replay scores, tickets (SQLite, CSV, one JSON per work order), drift, RESULTS.md      mlflow/  tracking + registry
```
