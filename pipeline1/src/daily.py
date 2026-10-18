"""One row per calendar day: what the day asked of me, plus the next morning's recovery."""
import numpy as np
import pandas as pd
from features import TZ, TAG_PRIORITY, VALENCE
from recovery import morning_signals, readiness, _read
from stage_b import window_residuals, daily_strain

COGNITIVE = ["class_cs156", "class_other", "study", "work", "high_stress_task", "meeting"]


def _by_day(x, col="value", how="sum"):
    return getattr(x.groupby(x.startDate.dt.strftime("%Y-%m-%d"))[col], how)()


def day_load(w, watch_dir, hi_hr):
    d = pd.DataFrame(index=sorted(w.date.unique()))
    steps = _read(f"{watch_dir}/steps.csv")
    d["steps"] = _by_day(steps).reindex(d.index.strftime("%Y-%m-%d")).values
    for name, f in [("exercise_min", "exercise_min"), ("daylight_min", "daylight_min")]:
        d[name] = _by_day(_read(f"{watch_dir}/{f}.csv")).reindex(d.index.strftime("%Y-%m-%d")).values
    noise = _read(f"{watch_dir}/noise_db.csv")
    d["noise_db"] = _by_day(noise, how="mean").reindex(d.index.strftime("%Y-%m-%d")).values
    eff = _read(f"{watch_dir}/physical_effort.csv")
    eff["min"] = (eff.endDate - eff.startDate).dt.total_seconds() / 60
    hi = eff[eff.value >= eff.value.quantile(0.80)]
    d["effort_hi_min"] = _by_day(hi, "min").reindex(d.index.strftime("%Y-%m-%d")).values
    late = hi[hi.startDate.dt.hour >= 21]
    d["effort_late_min"] = _by_day(late, "min").reindex(d.index.strftime("%Y-%m-%d")).values
    d = d.fillna({"steps": 0, "exercise_min": 0, "daylight_min": 0, "effort_hi_min": 0, "effort_late_min": 0})
    awake = w[~w.asleep]
    for tag in TAG_PRIORITY:
        d[f"h_{tag}"] = (awake[f"cal_{tag}"].groupby(awake.date).sum() * 5 / 60).reindex(d.index).fillna(0)
    d["cognitive_h"] = d[[f"h_{t}" for t in COGNITIVE]].sum(axis=1)
    d["costly_h"] = d[[f"h_{t}" for t in VALENCE["costly"]]].sum(axis=1)
    d["refreshing_h"] = d[[f"h_{t}" for t in VALENCE["refreshing"]]].sum(axis=1)
    d["unwell"] = awake.cal_unwell.groupby(awake.date).any().reindex(d.index).fillna(False)
    d["place"] = awake.place.groupby(awake.date).agg(lambda s: s.mode().iat[0]).reindex(d.index)
    d["weekend"] = d.index.dayofweek >= 5
    return d


def build_daily(w, watch_dir):
    a, info = window_residuals(w)
    strain = daily_strain(a, info["hi_hr"])
    load = day_load(w, watch_dir, info["hi_hr"])
    mor = readiness(morning_signals(watch_dir))
    d = load.join(strain)
    d["refreshing_h"] = d.refreshing_h + d.dance_min.fillna(0) / 60            # dancing counts as refreshing
    d = d.join(mor.add_prefix("m_").shift(-1, freq="D"))    # tomorrow's morning sits on today's row (m_ = next morning)
    d = d.join(mor[["asleep_h", "composite", "score", "score_lo", "score_hi"]].add_prefix("this_"))  # this morning (before the day)
    ok = d.windows >= 60                                        # days with too little watch time carry no day features
    d.loc[~ok.fillna(False), ["S", "M", "P_min"]] = np.nan
    return d, a, info, mor
