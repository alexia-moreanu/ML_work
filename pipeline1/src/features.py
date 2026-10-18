"""Step 2: watch CSVs + calendar tags -> one row per 5-minute window (Stage A table)."""
import numpy as np
import pandas as pd

import os
TZ = os.environ.get("HEALTH_TZ", "Europe/Madrid")   # set HEALTH_TZ to your own time zone
WIN = "5min"
# when tags overlap, the first one in this list wins
TAG_PRIORITY = ["class_cs156", "class_other", "high_stress_task", "meeting", "study", "work", "social_late", "social",
                "nap", "travel", "exercise", "walk", "rest"]
# Good vs costly load. The watch sees the same racing heart in both, so this comes from the person: edit it to fit your own life.
# (Mine: friends, dancing and walking are refreshing; class, work and travelling are costly for my body.)
VALENCE = dict(costly=["class_cs156", "class_other", "high_stress_task", "meeting", "work", "travel"],
               refreshing=["social", "exercise", "walk"])
REFRESHING_WORKOUTS = ("Dance",)
STATE_TAGS = ["unwell"]                 # background states, can overlap any activity
PLACES = ["berlin", "uk", "madrid"]
ASLEEP = ("Core", "Deep", "REM", "Unspecified")


def _read(path, cols=("startDate", "endDate", "value")):
    d = pd.read_csv(path, usecols=list(cols))
    for c in ("startDate", "endDate"):
        d[c] = pd.to_datetime(d[c], utc=True).dt.tz_convert(TZ)
    return d


def load_sleep(watch_dir):
    s = _read(f"{watch_dir}/sleep.csv")
    s = s[s.value.str.contains("|".join(ASLEEP))].sort_values("startDate")
    # a "night" = asleep records chained with gaps < 2 h
    gap = s.startDate - s.endDate.shift()
    s["night_id"] = (gap > pd.Timedelta("2h")).cumsum()
    nights = s.groupby("night_id").agg(start=("startDate", "min"), end=("endDate", "max"))
    asleep_min = s.assign(m=(s.endDate - s.startDate).dt.total_seconds() / 60).groupby("night_id").m.sum()
    nights["asleep_min"] = asleep_min
    nights["wake_date"] = nights.end.dt.normalize()
    return nights.reset_index(drop=True)


def load_calendar(path):
    c = pd.read_csv(path)
    for k in ("start", "end"):
        c[k] = pd.to_datetime(c[k]).dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    return c


def build_windows(watch_dir, cal_path):
    hr = _read(f"{watch_dir}/heart_rate.csv")
    hr["t"] = hr.startDate.dt.floor(WIN)
    w = hr.groupby("t").agg(hr=("value", "mean"), hr_n=("value", "size"))

    st = _read(f"{watch_dir}/steps.csv")
    st["t"] = st.startDate.dt.floor(WIN)
    ae = _read(f"{watch_dir}/active_energy.csv")
    ae["t"] = ae.startDate.dt.floor(WIN)

    idx = pd.date_range(w.index.min(), w.index.max(), freq=WIN)
    w = w.reindex(idx)
    w.index.name = "t"
    w["steps"] = st.groupby("t").value.sum().reindex(idx).fillna(0)
    w["active_kcal"] = ae.groupby("t").value.sum().reindex(idx).fillna(0)

    # watch-only context signals (optional: skipped if the file is missing)
    for name, col in (("physical_effort", "effort"), ("noise_db", "noise_db")):
        try:
            x = _read(f"{watch_dir}/{name}.csv")
            x["t"] = x.startDate.dt.floor(WIN)
            w[col] = x.groupby("t").value.mean().reindex(idx)
        except FileNotFoundError:
            w[col] = np.nan
    w["noise_db"] = w.noise_db.ffill(limit=6)          # a noise reading stays valid for ~30 minutes
    w["effort"] = w.effort.fillna(w.effort.median())

    # lags: what the body did in the previous 5/10/15 min, and the last 30 min in total
    for k in (1, 2, 3):
        w[f"steps_l{k}"] = w.steps.shift(k).fillna(0)
    w["steps_30"] = w.steps.rolling(6, min_periods=1).sum()
    w["kcal_l1"] = w.active_kcal.shift(1).fillna(0)
    w["kcal_30"] = w.active_kcal.rolling(6, min_periods=1).sum()

    # sleep mask + time since waking
    nights = load_sleep(watch_dir)
    w["asleep"] = False
    w["mins_since_wake"] = np.nan
    for _, n in nights.iterrows():
        w.loc[(w.index >= n.start) & (w.index < n.end), "asleep"] = True
    for i, n in nights.iterrows():
        nxt = nights.start.iloc[i + 1] if i + 1 < len(nights) else w.index.max() + pd.Timedelta(WIN)
        m = (w.index >= n.end) & (w.index < nxt)
        w.loc[m, "mins_since_wake"] = (w.index[m] - n.end).total_seconds() / 60

    # clock
    h = w.index.hour + w.index.minute / 60
    w["hour_sin"], w["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    w["date"] = w.index.tz_localize(None).normalize()

    # workouts and calendar tags
    wk = _read(f"{watch_dir}/workouts.csv", ("workoutActivityType", "startDate", "endDate"))
    w["workout"] = ""
    for _, r in wk.iterrows():
        w.loc[(w.index >= r.startDate.floor(WIN)) & (w.index < r.endDate), "workout"] = r.workoutActivityType.replace("HKWorkoutActivityType", "")
    cal = load_calendar(cal_path)
    if "kind" not in cal:
        cal["kind"] = "activity"
    for tag in TAG_PRIORITY + STATE_TAGS:
        w[f"cal_{tag}"] = False
        for _, r in cal[cal.tag == tag].iterrows():
            w.loc[(w.index >= r.start) & (w.index < r.end), f"cal_{tag}"] = True
    w["cal_tag"] = "none"
    for tag in reversed(TAG_PRIORITY):            # first in TAG_PRIORITY wins when events overlap
        w.loc[w[f"cal_{tag}"], "cal_tag"] = tag
    # how it felt: the stress rating (1-5) I gave each activity; where events overlap the highest one counts
    w["felt_stress"] = np.nan
    if "felt_stress" in cal:
        rated = cal[(cal.kind == "activity") & cal.felt_stress.notna()]
        for _, r in rated.iterrows():
            m = (w.index >= r.start) & (w.index < r.end)
            w.loc[m, "felt_stress"] = np.fmax(w.loc[m, "felt_stress"], float(r.felt_stress))
    w["place"] = "unknown"
    for _, r in cal[cal.kind == "place"].iterrows():
        w.loc[(w.index >= r.start) & (w.index < r.end), "place"] = r.tag
    w["in_workout"] = w.workout != ""
    return w, nights


if __name__ == "__main__":
    import sys
    w, nights = build_windows(sys.argv[1], sys.argv[2])
    w.to_csv(sys.argv[3])
    print(nights)
