"""Fake watch data in the same CSV schema as parse_health.py, so anyone can try the pipeline without health data.

A hidden "stress" level per day lifts heart rate while sitting in class, and lowers that night's HRV and sleep.
It is only a toy world: use it to check that the code runs and that the pipeline can find a planted effect.
"""
import os
import numpy as np
import pandas as pd

FIELDS = ["startDate", "endDate", "value", "unit", "sourceName"]
SRC = "Synthetic Watch"
fmt = lambda t: t.strftime("%Y-%m-%d %H:%M:%S +0200")


def make(out_dir, days=21, start="2026-09-21", seed=156):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    rows = {k: [] for k in ["heart_rate", "steps", "active_energy", "sleep", "hrv", "resting_hr", "resp_rate", "wrist_temp",
                            "physical_effort", "exercise_min", "daylight_min", "noise_db", "workouts", "mindful", "stand_hour", "walking_hr", "vo2max"]}
    cal = []
    stress = rng.normal(0, 1, days + 1)
    d0 = pd.Timestamp(start)
    add = lambda k, s, e, v, u="": rows[k].append([fmt(s), fmt(e), v, u, SRC])
    for i in range(days):
        day = d0 + pd.Timedelta(days=i)
        wake = day + pd.Timedelta(hours=7 + rng.normal(0, 0.4))
        bed = day + pd.Timedelta(hours=23 + rng.normal(0, 0.7))
        class_start = day + pd.Timedelta(hours=15)
        dance = (i % 3 == 1)
        # hidden activity schedule in 5-minute steps
        t, step_hr = wake, []
        while t < bed:
            hour = t.hour + t.minute / 60
            in_class = (15 <= hour < 16.5) and day.dayofweek < 5
            walking = rng.random() < (0.12 if not in_class else 0.0) or (8 <= hour < 8.5)
            dancing = dance and 20 <= hour < 21.5
            steps = rng.poisson(380) if walking else (rng.poisson(40) if dancing else 0)
            hr = 64 + 3 * np.sin(2 * np.pi * (hour - 14) / 24) + 0.05 * steps + (38 if dancing else 0) \
                + (6 + 3 * stress[i] if in_class else 0) + max(stress[i], 0) * 1.5 + rng.normal(0, 3)
            n_reads = 3 if (walking or dancing) else 1
            for _ in range(n_reads):
                tt = t + pd.Timedelta(seconds=int(rng.integers(0, 300)))
                add("heart_rate", tt, tt, round(hr + rng.normal(0, 2), 0), "count/min")
            if steps:
                add("steps", t, t + pd.Timedelta(minutes=1), steps, "count")
                add("active_energy", t, t + pd.Timedelta(minutes=1), round(steps * 0.04, 2), "kcal")
            add("physical_effort", t, t + pd.Timedelta(minutes=5), round(2 + steps / 120 + (4 if dancing else 0), 1), "kcal/hr·kg")
            add("noise_db", t, t + pd.Timedelta(minutes=5), round(55 + rng.normal(0, 5), 1), "dBASPL")
            if hour in (10.0, 12.0) or (8 <= hour < 8.5 and t.minute == 0):
                add("daylight_min", t, t + pd.Timedelta(minutes=5), 5, "min")
            if walking or dancing:
                add("exercise_min", t, t + pd.Timedelta(minutes=5), 5, "min")
            t += pd.Timedelta(minutes=5)
        # one calendar label per day (class) and a dance workout
        if day.dayofweek < 5:
            cal.append([class_start, class_start + pd.Timedelta(minutes=90), "class_cs156", "activity"])
        if dance:
            s = day + pd.Timedelta(hours=20)
            rows["workouts"].append(["HKWorkoutActivityTypeCardioDance", fmt(s), fmt(s + pd.Timedelta(minutes=90)), 90, "min", SRC])
            cal.append([s, s + pd.Timedelta(minutes=90), "walk", "activity"])
        # the night after the day: stress lowers sleep and HRV
        sleep_h = np.clip(7.6 - 0.5 * max(stress[i], 0) + rng.normal(0, 0.5), 3.5, 9.5)
        s0, e0 = bed, bed + pd.Timedelta(hours=float(sleep_h))
        stages = ["Core", "Deep", "Core", "REM"] * 4
        cur = s0
        for j, stg in enumerate(stages):
            seg = (e0 - s0) / len(stages)
            add("sleep", cur, cur + seg, f"HKCategoryValueSleepAnalysisAsleep{stg}")
            cur += seg
        mid = s0 + (e0 - s0) / 2
        hrv = 58 - 7 * stress[i] + rng.normal(0, 6)
        add("hrv", mid, mid + pd.Timedelta(seconds=60), round(max(hrv, 15), 2), "ms")
        add("hrv", mid + pd.Timedelta(hours=1), mid + pd.Timedelta(hours=1, seconds=60), round(max(hrv + rng.normal(0, 4), 15), 2), "ms")
        add("resting_hr", e0, e0, round(58 + 2 * stress[i] + rng.normal(0, 1.5), 0), "count/min")
        add("resp_rate", mid, mid, round(15.5 + 0.4 * stress[i] + rng.normal(0, 0.4), 1), "count/min")
        add("wrist_temp", s0, e0, round(35.8 + rng.normal(0, 0.25), 3), "degC")
    for k, v in rows.items():
        cols = ["workoutActivityType", "startDate", "endDate", "duration", "durationUnit", "sourceName"] if k == "workouts" else FIELDS
        pd.DataFrame(v, columns=cols).to_csv(f"{out_dir}/{k}.csv", index=False)
    pl = pd.DataFrame(cal, columns=["start", "end", "tag", "kind"])
    pl = pd.concat([pl, pd.DataFrame([[d0, d0 + pd.Timedelta(days=days), "home", "place"]], columns=pl.columns)])
    pl.to_csv(os.path.join(os.path.dirname(out_dir.rstrip("/")), "calendar_tags.csv"), index=False)
    return out_dir
