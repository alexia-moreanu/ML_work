"""Ridge regression on the daily table, written from the closed form and checked against scikit-learn.

   beta_hat = (X'X + lambda*I)^-1 X'y          (X standardised, y centred)

Ridge is session 15 material (not yet covered). With ~9 usable days and ~10 features, plain least squares
cannot be solved reliably (collinearity, session 3), and lambda is a dial between bias and variance (session 11).
"""
import numpy as np
import pandas as pd

FEATS = ["S", "steps_k", "exercise_min", "daylight_min", "noise_db", "effort_late_min", "cognitive_h", "h_social_all",
         "h_travel", "this_asleep_h"]
LAMBDAS = np.logspace(-2, 3, 21)


def prepare(d):
    x = d.copy()
    x["steps_k"] = x.steps / 1000
    x["h_social_all"] = x.h_social + x.h_social_late
    return x


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z, yc = (X - mu) / sd, y - y.mean()
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ yc)
    return dict(beta=beta, mu=mu, sd=sd, ybar=y.mean())


def ridge_predict(m, X):
    return ((X - m["mu"]) / m["sd"]) @ m["beta"] + m["ybar"]


def usable(d, target, feats=FEATS):
    x = prepare(d)
    return x.dropna(subset=feats + [target])


def lodo_curve(D, feats, target, lams=LAMBDAS):
    """Leave-one-day-out MSE for every lambda (and training MSE for the bias-variance picture)."""
    X, y = D[feats].values.astype(float), D[target].values.astype(float)
    rows = []
    for lam in lams:
        cv, tr = [], []
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            m = ridge_fit(X[keep], y[keep], lam)
            cv.append((y[i] - ridge_predict(m, X[i:i + 1])[0]) ** 2)
            tr.append(np.mean((y[keep] - ridge_predict(m, X[keep])) ** 2))
        rows.append(dict(lam=lam, cv_mse=np.mean(cv), train_mse=np.mean(tr)))
    return pd.DataFrame(rows)


def mean_baseline_mse(y):
    return np.mean([(y[i] - np.delete(y, i).mean()) ** 2 for i in range(len(y))])


def permutation_p(D, feats, target, n=1000, seed=156):
    """Does the best-lambda model beat predicting the mean more often than shuffled targets would?
    The lambda search is repeated inside every shuffle, so choosing lambda cannot flatter the result."""
    rng = np.random.default_rng(seed)
    X, y = D[feats].values.astype(float), D[target].values.astype(float)

    def gain(yv):
        Dp = pd.DataFrame(X, columns=feats); Dp[target] = yv
        best = lodo_curve(Dp, feats, target, LAMBDAS[::2]).cv_mse.min()
        return 1 - best / mean_baseline_mse(yv)
    real = gain(y)
    null = np.array([gain(rng.permutation(y)) for _ in range(n)])
    return real, (np.sum(null >= real) + 1) / (n + 1), null


def knn_lodo_mse(D, feats, target, k=3):
    """k-nearest-neighbour 'analog days' (not covered in class): predict a day's next morning from the k most similar other days."""
    X, y = D[feats].values.astype(float), D[target].values.astype(float)
    errs = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        mu, sd = X[keep].mean(0), np.where(X[keep].std(0) > 0, X[keep].std(0), 1)
        dist = np.linalg.norm((X[keep] - mu) / sd - (X[i] - mu) / sd, axis=1)
        errs.append((y[i] - y[keep][np.argsort(dist)[:k]].mean()) ** 2)
    return float(np.mean(errs))


def persistence_mse(D, today, target):
    """Naive rival: tomorrow's morning will look like this morning."""
    return float(np.mean((D[target] - D[today]) ** 2))
