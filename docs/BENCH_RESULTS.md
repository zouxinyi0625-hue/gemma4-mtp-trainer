# Bench Results

vLLM online bench (`vllm-msn/benchmarks/gemma4_12b_fp8/run_maiprofile_online.sh`) is
the **only authoritative metric** for this project. Training-log accept numbers
(teacher-forcing coordinate) do **not** track deployment accept — several times the
training log improved while the bench regressed. Only trust the numbers below.

## Setup

- config `26b_e011_mtp.json`, 26B-A4B FP8 target, **TP1** (TP2 triggers a Marlin FP8
  MoE `Invalid thread config` kernel crash — the sharded MoE dim has no matching
  tile; TP1 does not shard and works).
- 200 prompts/layer, new eval split (`eval_maiprofile_26b.jsonl` →
  `convert_new_eval_to_sc1.py`), `SKIP_CONVERT=1`.
- Same 200 prompts across runs (`--seed` defaults to 0, CustomDataset seeds then
  shuffles then takes first N). Stock vs trained differ **only** in
  `GEMMA4_ASSISTANT_MODEL_PATH`.

## Bench #3 — trained vs stock (2026-07-22, new eval)

Trained = off-by-one fix + expanded data (226k train) + sequential/shuffled cache +
1 epoch. **Result: every layer and every position regressed vs stock, by 10–15
points.** Worse than bench #2 (which was 3–10 below stock).

| layer                        | stock accept% | trained accept% |    Δ   |
|------------------------------|:-------------:|:---------------:|:------:|
| layer3_seasonality           |     99.06     |      97.32      | −1.74  |
| layer3_commercial_interests  |     82.05     |      68.31      | −13.74 |
| layer1_actual                |     77.42     |      62.00      | −15.42 |
| layer4_commercial_preference |     68.16     |      53.13      | −15.03 |
| layer1_delta                 |     65.65     |      53.24      | −12.41 |
| layer1_intent                |     58.53     |      44.44      | −14.09 |
| layer2_coarse_interest       |     56.76     |      45.86      | −10.90 |
| layer3_persona               |     52.90     |      39.33      | −13.57 |
| layer2_temporal              |     52.58     |      40.40      | −12.18 |
| layer4_biography             |     36.59     |      27.92      | −8.67  |

### Per-position (pos0 = step0) — the decisive signal

**pos0 regressed on every layer too.** After the off-by-one fix, step0 input and
supervision match vLLM exactly, so pos0 should ≈ stock. It dropped → training
damages the weights **from the very first step**. This **rules out the
teacher-forcing gap** as the primary cause (that only affects step ≥ 1). Root cause
is upstream: the training forward / objective diverges from the vLLM single-anchor
deployment forward from step0 onward.

| layer                        | stock pos0 | trained pos0 |
|------------------------------|:----------:|:------------:|
| layer1_actual                |    90.12   |    83.27     |
| layer3_commercial_interests  |    93.71   |    86.84     |
| layer4_commercial_preference |    85.01   |    77.68     |
| layer1_delta                 |    86.16   |    83.07     |
| layer1_intent                |    80.03   |    73.19     |
| layer4_biography             |    68.06   |    62.13     |

Trained per-position (accept%):

```
layer1_actual                pos0=83.27 pos1=70.42 pos2=59.76 pos3=51.95 pos4=44.61
layer1_delta                 pos0=83.07 pos1=67.44 pos2=53.39 pos3=37.49 pos4=24.79
layer1_intent                pos0=73.19 pos1=52.18 pos2=39.98 pos3=32.94 pos4=23.89
layer2_coarse_interest       pos0=73.16 pos1=53.08 pos2=41.89 pos3=34.30 pos4=26.86
layer2_temporal              pos0=69.93 pos1=48.37 pos2=36.51 pos3=28.04 pos4=19.16
layer3_commercial_interests  pos0=86.84 pos1=77.46 pos2=67.94 pos3=59.86 pos4=49.45
layer3_persona               pos0=69.48 pos1=47.95 pos2=34.49 pos3=26.79 pos4=17.93
layer3_seasonality           pos0=99.22 pos1=98.84 pos2=97.94 pos3=96.41 pos4=94.22
layer4_biography             pos0=62.13 pos1=36.36 pos2=21.09 pos3=12.54 pos4=7.47
layer4_commercial_preference pos0=77.68 pos1=62.18 pos2=50.67 pos3=40.73 pos4=34.39
```

## Bench #4 — uniform training (2026-07-24)

Uniform training run, e011 MTP layer4_commercial_preference.
Result file: `online_results/26b_e011_mtp_layer4_commercial_preference_online_20260724_015941.txt`
Summary JSON: `maiprofile_26b_e011_mtp_online_summary.json`

| layer                        | accept% | accept_len | out_tok/s |
|------------------------------|:-------:|:----------:|:---------:|
| layer3_seasonality           |  97.86  |    5.89    |  4859.3   |
| layer3_commercial_interests  |  67.67  |    4.38    |  1940.6   |
| layer1_actual                |  63.36  |    4.17    |  2108.1   |
| layer4_commercial_preference |  55.25  |    3.76    |  2162.3   |
| layer1_delta                 |  53.07  |    3.65    |  1183.9   |
| layer2_coarse_interest       |  45.26  |    3.26    |  1808.8   |
| layer1_intent                |  44.28  |    3.21    |  1924.1   |
| layer2_temporal              |  40.41  |    3.02    |  1676.2   |
| layer3_persona               |  38.52  |    2.93    |  1503.9   |
| layer4_biography             |  29.03  |    2.45    |   225.6   |

Per-position acceptance (%):

```
layer1_actual                pos0=84.59 pos1=71.78 pos2=61.18 pos3=53.36 pos4=45.87
layer1_delta                 pos0=83.94 pos1=67.01 pos2=53.02 pos3=37.29 pos4=24.08
layer1_intent                pos0=72.88 pos1=52.39 pos2=39.77 pos3=32.40 pos4=23.94
layer2_coarse_interest       pos0=71.46 pos1=52.98 pos2=41.76 pos3=33.45 pos4=26.64
layer2_temporal              pos0=70.08 pos1=48.98 pos2=36.86 pos3=27.51 pos4=18.63
layer3_commercial_interests  pos0=87.25 pos1=76.50 pos2=67.17 pos3=58.12 pos4=49.34
layer3_persona               pos0=69.25 pos1=47.32 pos2=33.51 pos3=25.72 pos4=16.77
layer3_seasonality           pos0=99.22 pos1=98.84 pos2=98.32 pos3=97.09 pos4=95.85
layer4_biography             pos0=63.61 pos1=37.81 pos2=22.72 pos3=13.45 pos4=7.56
layer4_commercial_preference pos0=79.45 pos1=64.06 pos2=52.66 pos3=42.90 pos4=37.20
```

## History (old eval split — NOT comparable to above; different dataset)

| layer            | stock | #1 bug | #2 off-by-one fixed |
|------------------|:-----:|:------:|:-------------------:|
| layer1_actual    | 75.56 | 55.83  |        64.81        |
| layer1_intent    | 59.32 | 38.21  |        49.41        |
| layer2_temporal  | 54.56 | 35.59  |        45.53        |
| layer3_seasonality | 98.99 | 91.21 |        95.87        |
| layer4_commercial | 68.72 | 53.88  |        65.48        |

- **#1 (bug)**: off-by-one token/position error — training was one step ahead of
  inference. Fixed in commit 89175e5.
- **#2 (fixed)**: off-by-one fix pulled the 15–21pt collapse back to 3–10pt below
  stock, but still below stock.

## Open problem

Training regresses the stock weights from step0. Do **not** keep tuning data
volume / epochs / lr / cache order — three benches have shown these don't fix it.
Next step: dump step0 draft logits from the **training forward** and the **vLLM
forward** for the **same prompt under stock weights**, and compare element-wise. If
they already disagree under stock weights, the training forward has a bug
(numerics, mask, position, embed norm, KV concat, or the argmax-CE target itself)
that pushes the weights in the wrong direction.
