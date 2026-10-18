"""Stage A: how much heart rate does movement explain? Models, feature sets, CV schemes, baselines."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 156
from features import TAG_PRIORITY, STATE_TAGS
CAL = [f"cal_{t}" for t in TAG_PRIORITY + STATE_TAGS]
FEATURES = {
    "steps only": ["steps"],
    "+ step lags": ["steps", "steps_l1", "steps_l2", "steps_l3", "steps_30"],
    "+ lags & energy": ["steps", "steps_l1", "steps_l2", "steps_l3", "steps_30", "active_kcal", "kcal_l1", "kcal_30"],
    "+ clock, wake, calendar": ["steps", "steps_l1", "steps_l2", "steps_l3", "steps_30", "active_kcal", "kcal_l1", "kcal_30",
                                "hour_sin", "hour_cos", "mins_since_wake"] + CAL,
}


def awake_table(w):
    a = w[(~w.asleep) & w.hr.notna()].copy()
    a["mins_since_wake"] = a.mins_since_wake.fillna(a.mins_since_wake.median())
    a["workout"] = a.workout.fillna("")
    for c in CAL + ["in_workout"]:
        a[c] = a[c].astype(float)
    return a


def make_model(name):
    if name == "Linear regression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Random forest":
        return RandomForestRegressor(300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    if name == "Neural net (16-8)":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16, 8), alpha=1e-2, max_iter=3000, random_state=SEED))
    raise ValueError(name)


MODELS = ["Linear regression", "Random forest", "Neural net (16-8)"]


def splitter(kind, a):
    if kind == "random 5-fold":
        return KFold(5, shuffle=True, random_state=SEED).split(a)
    return LeaveOneGroupOut().split(a, groups=a.date)  # "leave one day out"


def hour_mean_baseline(a, tr, te):
    m = a.iloc[tr].groupby(a.index[tr].hour).hr.mean()
    return a.index[te].hour.map(m).fillna(a.iloc[tr].hr.mean()).values


def oof(a, model, feats, kind):
    """Out-of-fold predictions. model='hour-mean' or 'global mean' are the baselines."""
    pred = np.full(len(a), np.nan)
    for tr, te in splitter(kind, a):
        if model == "hour-mean":
            pred[te] = hour_mean_baseline(a, tr, te)
        elif model == "global mean":
            pred[te] = a.hr.iloc[tr].mean()
        else:
            m = make_model(model).fit(a.iloc[tr][feats], a.hr.iloc[tr])
            pred[te] = m.predict(a.iloc[te][feats])
    return pred


def scores(y, p):
    r = y - p
    return dict(MAE=np.abs(r).mean(), RMSE=np.sqrt((r ** 2).mean()), R2=1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum())


def run_grid(a):
    rows = []
    for kind in ("random 5-fold", "leave-one-day-out"):
        for b in ("global mean", "hour-mean"):
            rows.append(dict(split=kind, features="-", model=b, **scores(a.hr.values, oof(a, b, None, kind))))
        for fname, feats in FEATURES.items():
            for mname in MODELS:
                rows.append(dict(split=kind, features=fname, model=mname, **scores(a.hr.values, oof(a, mname, feats, kind))))
    return pd.DataFrame(rows)


def uhr(a, pred, sigma=None, tau=1.5):
    """z_t = (HR - HR_hat)/sigma_r ; UHR_t = mean of z over this and previous 2 consecutive windows."""
    r = a.hr.values - pred
    sigma = np.sqrt((r ** 2).mean()) if sigma is None else sigma   # MLE of a zero-mean Gaussian: 1/N
    z = pd.Series(r / sigma, index=a.index)
    out = z.copy()
    for k in (1, 2):
        shifted = z.shift(k, freq="5min").reindex(z.index)
        out = out + shifted
    n = 1 + sum(z.shift(k, freq="5min").reindex(z.index).notna().astype(int) for k in (1, 2))
    return (out / n), z, sigma


if __name__ == "__main__":
    w = pd.read_csv("data_private/windows.csv", index_col=0, parse_dates=True)
    w.index = pd.to_datetime(w.index, utc=True).tz_convert("Europe/Madrid")
    w["date"] = pd.to_datetime(w["date"])
    a = awake_table(w)
    g = run_grid(a)
    pd.set_option("display.width", 200)
    print(g.round(3).to_string())


def learning_curve(a, feats, models=MODELS, ks=(1, 2, 3, 4, 6, 8, 10), reps=6):
    """Train on k random other days, test on one held-out day. Mean MAE over held-out days and repeats."""
    rng = np.random.default_rng(SEED)
    days = np.array(sorted(a.date.unique()))
    rows = []
    for k in ks:
        for _ in range(reps):
            te_day = rng.choice(days)
            pool = days[days != te_day]
            tr_days = rng.choice(pool, size=min(k, len(pool)), replace=False)
            tr, te = a[a.date.isin(tr_days)], a[a.date == te_day]
            for mname in models:
                m = make_model(mname).fit(tr[feats], tr.hr)
                rows.append(dict(k=k, model=mname, MAE=np.abs(te.hr - m.predict(te[feats])).mean(),
                                 train_MAE=np.abs(tr.hr - m.predict(tr[feats])).mean()))
    return pd.DataFrame(rows).groupby(["model", "k"]).mean().reset_index()
