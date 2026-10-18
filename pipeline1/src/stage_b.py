"""Stage B building blocks: movement-adjusted heart-rate residuals (UHR) and daily strain.

Stage A predicts HR from movement. Here its leave-one-day-out residual becomes
  z_t   = (HR_t - HRhat_t) / sigma_r          (sigma_r from sitting windows, MLE with 1/N)
  UHR_t = mean(z_t, z_{t-1}, z_{t-2})         (unexplained heart rate, 15-minute memory)
and the day is summarised by
  M_d = share of sitting windows with UHR > tau,   P_d = minutes in my own top-20 % HR zone,
  S_d = z(M_d) + z(P_d).
"""
import numpy as np
import pandas as pd
from stage_a import FEATURES, awake_table, oof, uhr

TAU = 1.5          # UHR threshold, in residual standard deviations
MIN_SED = 20       # sitting windows needed before M_d is trusted
STRAIN_FEATS = FEATURES["+ step lags"]   # no active energy: the watch computes it partly from heart rate


def zscore(x):
    return (x - x.mean()) / x.std(ddof=0)


def window_residuals(w, model="Neural net (16-8)"):
    """Leave-one-day-out residuals for every awake window. Returns (windows, info)."""
    a = awake_table(w)
    pred = oof(a, model, STRAIN_FEATS, "leave-one-day-out")
    is_sed = ((a.steps + a.steps_l1 + a.steps_l2) == 0) & (~a.in_workout.astype(bool))   # no steps for 15 min
    sigma = np.sqrt(((a.hr.values - pred)[is_sed.values] ** 2).mean())
    u, z, sigma = uhr(a, pred, sigma=sigma)
    a = a.assign(pred=pred, z=z, uhr=u, sedentary=is_sed)
    return a, dict(sigma=float(sigma), hi_hr=float(a.hr.quantile(0.80)), model=model)


def daily_strain(a, hi_hr):
    sed = a[a.sedentary]
    d = pd.DataFrame({
        "windows": a.groupby("date").size(),
        "sed_windows": sed.groupby("date").size(),
        "M": sed.assign(x=sed.uhr > TAU).groupby("date").x.mean(),
        "P_min": a.assign(x=a.hr >= hi_hr).groupby("date").x.sum() * 5,
        "dance_min": a[a.workout.str.contains("Dance")].groupby("date").size() * 5,
    }).fillna({"dance_min": 0})
    d.loc[d.sed_windows.fillna(0) < MIN_SED, "M"] = np.nan
    d["S"] = zscore(d.M) + zscore(d.P_min)
    return d
