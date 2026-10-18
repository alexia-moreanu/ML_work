"""Kinds of moments (and days) without labels: PCA to compress, k-means to group.

Every awake 5-minute window becomes a point described by heart rate, the movement-adjusted residual z, recent steps,
physical effort, noise and time of day. PCA (session 16) finds the few directions that carry most of the variance,
k-means groups the points, and a day is then described by how its time is spread over the groups (a "day fingerprint").
The calendar labels are NOT used to build the groups; they are only used afterwards to ask what each group is.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

MOMENT_FEATS = ["hr", "z", "steps_30", "effort", "noise_db", "hour_sin", "hour_cos"]
SEED = 156


def moment_matrix(a):
    X = a[MOMENT_FEATS].copy()
    X["steps_30"] = np.log1p(X.steps_30)
    X["noise_db"] = X.noise_db.fillna(X.noise_db.median())
    return X


def fit_moments(a, k=4, ks=range(2, 8), n_pc=3):
    X = StandardScaler().fit_transform(moment_matrix(a))
    pca = PCA(n_components=n_pc, random_state=SEED).fit(X)
    P = pca.transform(X)
    sil = {k: silhouette_score(P, KMeans(k, n_init=10, random_state=SEED).fit_predict(P)) for k in ks}
    # silhouette is flat across k (the windows form a continuum, not tight islands), so k is fixed up front for readability
    km = KMeans(k, n_init=10, random_state=SEED).fit(P)
    return dict(pca=pca, P=P, km=km, k=k, silhouette=sil, labels=km.labels_)


def describe(a, res):
    a = a.assign(moment=res["labels"])
    prof = a.groupby("moment").agg(windows=("hr", "size"), hr=("hr", "mean"), z=("z", "mean"), steps_30=("steps_30", "mean"),
                                   effort=("effort", "mean"), noise_db=("noise_db", "mean"), hour=("hour_sin", "size"))
    prof["hour"] = a.groupby("moment").apply(lambda g: g.index.hour.values.mean())
    return a, prof.round(2)


def fingerprints(a):
    """Share of awake windows each day spends in each kind of moment."""
    f = pd.crosstab(a.date, a.moment, normalize="index")
    f.columns = [f"m{c}" for c in f.columns]
    return f


def tag_agreement(a):
    return adjusted_mutual_info_score(a.cal_tag, a.moment)
