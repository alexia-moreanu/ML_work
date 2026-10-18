"""Morning recovery signals and the readiness score.

For every morning we take what the watch measured overnight: HRV, resting HR, respiratory rate,
sleeping wrist temperature and sleep length. Each one is compared with MY baseline
(mean and SD of my other days, MLE with 1/N) and the signed z-scores are averaged:

  z_k   = (x_k - mu_k) / sigma_k          (sign flipped when lower is better)
  C_m   = mean_k z_k                      composite for morning m
  score = 100 * Phi(C_m)                  0-100, 50 = a typical morning for me
"""
import numpy as np
import pandas as pd
from scipy import stats
from features import TZ, load_sleep

SIGNALS = {               # name: direction (+1 higher is better, -1 lower is better, 0 = deviation either way is worse)
    "hrv_night": +1, "rhr": -1, "resp_night": -1, "wrist_temp": 0, "asleep_h": +1,
}


def _read(path):
    x = pd.read_csv(path)
    for c in ("startDate", "endDate"):
        x[c] = pd.to_datetime(x[c], utc=True).dt.tz_convert(TZ)
    return x


def overlap(x, n):
    """Records that span a stretch of time (the sleeping wrist temperature covers the night)."""
    return x[(x.startDate <= n.end) & (x.endDate >= n.start)].value


def morning_signals(watch_dir):
    """One row per wake-up date, from the longest sleep session ending that day."""
    nights = load_sleep(watch_dir)
    main = nights.sort_values("asleep_min").groupby(nights.wake_date.dt.strftime("%Y-%m-%d")).tail(1)
    hrv, rhr = _read(f"{watch_dir}/hrv.csv"), _read(f"{watch_dir}/resting_hr.csv")
    resp, temp = _read(f"{watch_dir}/resp_rate.csv"), _read(f"{watch_dir}/wrist_temp.csv")
    rows = []
    for _, n in main.iterrows():
        inside = lambda x: x[(x.startDate >= n.start) & (x.startDate <= n.end + pd.Timedelta("1h"))].value
        date = pd.Timestamp(n.wake_date.strftime("%Y-%m-%d"))
        r = rhr[rhr.startDate.dt.strftime("%Y-%m-%d") == date.strftime("%Y-%m-%d")].value
        rows.append(dict(date=date, hrv_night=np.log(inside(hrv)).mean() if len(inside(hrv)) else np.nan,
                         n_hrv=len(inside(hrv)), rhr=r.mean() if len(r) else np.nan,
                         resp_night=inside(resp).mean() if len(inside(resp)) else np.nan,
                         wrist_temp=overlap(temp, n).mean() if len(overlap(temp, n)) else np.nan,
                         asleep_h=n.asleep_min / 60, bedtime_h=(n.start.hour + n.start.minute / 60 + 12) % 24 - 12))
    return pd.DataFrame(rows).set_index("date").sort_index()


def _composite(row, mu, sd):
    z = []
    for k, sign in SIGNALS.items():
        if np.isnan(row[k]) or np.isnan(mu[k]) or not sd[k] > 0:
            continue
        v = (row[k] - mu[k]) / sd[k]
        z.append(sign * v if sign else -abs(v))
    return np.mean(z) if len(z) >= 3 else np.nan      # need at least 3 of the 5 signals


def readiness(m, boot=500, seed=156):
    """Leave-one-day-out baseline: a morning is never compared with a baseline that contains itself.
    The bootstrap resamples the baseline days, so the interval says how unsure the baseline is."""
    rng = np.random.default_rng(seed)
    out = []
    for d, row in m.iterrows():
        base = m.drop(index=d)
        mu, sd = base[list(SIGNALS)].mean(), base[list(SIGNALS)].std(ddof=0)
        c = _composite(row, mu, sd)
        bs = []
        for _ in range(boot):
            b = base.iloc[rng.integers(0, len(base), len(base))]
            bs.append(_composite(row, b[list(SIGNALS)].mean(), b[list(SIGNALS)].std(ddof=0)))
        lo, hi = np.nanpercentile(bs, [5, 95]) if not np.isnan(c) else (np.nan, np.nan)
        out.append(dict(date=d, composite=c, score=100 * stats.norm.cdf(c), score_lo=100 * stats.norm.cdf(lo),
                        score_hi=100 * stats.norm.cdf(hi)))
    return m.join(pd.DataFrame(out).set_index("date"))
