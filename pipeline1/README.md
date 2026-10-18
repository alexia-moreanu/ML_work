# watchlab: from Apple Watch data to a daily read

Code for turning an **Apple Watch** export (plus a labelled calendar) into:

- **how much of your heart rate is just movement** (linear regression, random forest, small neural network, validated by leaving out whole days),
- **calm / elevated states** through the day: heart rate that movement can't explain, smoothed with a hidden Markov model,
- a **morning readiness score**, like a Garmin's, built from *your own* baseline,
- a **weekend recovery plan** based on the week you just had, where every line says where it comes from.

It was built for the CS156 First Pipeline assignment on 13 days of my own data. The full write-up, with every result and figure, is
[`notebooks/first_pipeline.ipynb`](notebooks/first_pipeline.ipynb). My raw health data is **not** in this repository; the `demo_data/` folder is synthetic.

---

## 1. Install

```bash
git clone https://github.com/alexia-moreanu/ML_work.git
cd ML_work/pipeline1
pip install -e .            # installs the `watchlab` command and its dependencies
```

Python 3.9+ works. To also run the notebook and the tests: `pip install -e ".[dev]"`.

## 2. Try it without any health data

```bash
watchlab demo               # makes 21 days of fake watch data + calendar, runs everything, prints a weekend plan
pytest -q tests             # 5 checks: ridge vs scikit-learn, HMM forward pass vs hmmlearn, the MLE sigma,
                            # the watch-only parser, and a planted effect in the fake data
```

## 3. Use it on your own watch

**Step 1: export your data.** On your iPhone: Health app → your profile picture → **Export All Health Data**. AirDrop or email the zip to your computer and unzip it. You get a folder with `export.xml` inside.

**Step 2: keep only the watch records.** Your phone also counts steps, so keeping both would double count every walk.

```bash
watchlab parse path/to/apple_health_export/export.xml data_private/watch
```

**Step 3: label your calendar.** Create `data_private/calendar_tags.csv`, one row per event:

| column | required | example | meaning |
|:--|:--|:--|:--|
| `start`, `end` | yes | `2026-10-08 15:00` | local time of the event |
| `tag` | yes | `class_cs156`, `study`, `walk`, `social`, `unwell` | what you were doing |
| `kind` | yes | `activity` / `state` / `place` | `state` = a background condition that can overlap anything (like being sick) |
| `felt_stress` | no | `1` to `5` | how stressful it felt; used to check the model against how you felt |
| `intensity`, `note`, `source` | no | `dreaded`, free text, `user` | extra context |

The tags are yours to define. Open `src/features.py` and edit:

- `TAG_PRIORITY`: your tags, in the order that wins when two events overlap,
- `VALENCE`: which activities are **costly** and which are **refreshing** *for you* (the watch sees the same fast heart rate in both, so only you can say which is which),
- `TZ`, or set the environment variable: `export HEALTH_TZ=Europe/Berlin`.

**Step 4: run it.**

```bash
watchlab run data_private/watch data_private/calendar_tags.csv --out outputs_run
```

This writes the readiness table and `weekend_plan.md` to `outputs_run/` and prints the plan.

**Step 5 (optional): the full notebook.** `notebooks/first_pipeline.ipynb` reads `../data_private/watch` and `../data_private/calendar_tags.csv`, so if you followed the steps above it runs as is. To run it on the synthetic data instead, change the paths in the first code cell to `../demo_data/watch` and `../demo_data/calendar_tags.csv`. Slow steps are cached in `data_private/cache/`, and the cache key changes whenever your data or the code changes, so it never serves stale results.

All thresholds for the weekend advice (sleep debt, what counts as a costly week, noise, ...) sit in one dictionary, `RULES` in `src/advice.py`, so you can change them to fit your life.

> **Never commit your export or anything made from it.** The repository's `.gitignore` already excludes `apple_health_export*/`, `pipeline1/data_private/` and `pipeline1/outputs/`.

---

## What is where

| File | What it does |
|:--|:--|
| `src/parse_health.py` | stream-parses `export.xml`, keeps watch records only |
| `src/features.py` | 5-minute windows: heart rate, steps and step lags, sleep, workouts, calendar tags |
| `src/stage_a.py` | Module A: predicts heart rate from movement, leave-one-day-out CV, learning curve |
| `src/stage_b.py` | Module B: residual z, unexplained heart rate (UHR), daily strain |
| `src/hmm_states.py` | Module B: Gaussian HMM with calm / elevated states, forward pass written by hand |
| `src/recovery.py` | Module C: overnight signals → readiness score with a bootstrap band |
| `src/daily.py`, `src/daily_models.py` | Module D1: one row per day; ridge, KNN, persistence, permutation test |
| `src/moments.py` | Module D2: PCA + k-means "kinds of moments" |
| `src/advice.py` | Module D3: weekend plan with labelled sources, all rules in `RULES` |
| `src/synthetic.py` | fake data with a planted effect, for the demo and the tests |
| `src/pipeline.py`, `src/cli.py` | runs everything end to end; the `watchlab` command |

## Limits

- It was built and tested on one person over 13 days, so treat it as a starting point you adapt to yourself.
- Small samples give wide uncertainty, and the notebook says where.
- "Elevated" is relative to *your own* average. A heart rate above prediction can be caffeine, excitement or illness as well as stress.
- The advice is general wellbeing guidance, not medical advice.

## License

MIT, see [`LICENSE`](../LICENSE).
