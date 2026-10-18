"""Calm / elevated states with a Gaussian hidden Markov model (session 22 material, not yet covered).

Each 5-minute window has one observation o_t = z_t (heart rate left over after movement). The model assumes
a hidden state s_t in {1..K} that I never observe:

   P(s_t = j | s_{t-1} = i) = A_ij                    (states are sticky: "elevated" tends to stay elevated)
   o_t | s_t = j  ~  N(mu_j, Sigma_j)

The forward recursion gives the likelihood of a whole day without summing over K^T paths:

   alpha_t(j) = [ sum_i alpha_{t-1}(i) A_ij ] * N(o_t | mu_j, Sigma_j),     P(o_1..T) = sum_j alpha_T(j)

Parameters come from EM (maximum likelihood, session 6, with hidden variables). K is chosen by held-out-day log-likelihood.
"""
import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

MAX_GAP = 2          # up to one missing 5-minute window is bridged; a longer gap starts a new sequence


def sequences(a, max_gap=MAX_GAP):
    """Cut the awake, non-workout windows into runs without long holes. Returns (obs array, lengths, run ids)."""
    a = a[~a.in_workout.astype(bool)].copy()
    a["steps15"] = a.steps + a.steps_l1 + a.steps_l2
    gap = np.r_[np.inf, np.diff(a.index.values) / np.timedelta64(5, "m")]
    a["run"] = np.cumsum(gap > max_gap)
    X = a.z.values[:, None]          # the residual is already movement-adjusted, so states mean 'calm' / 'elevated', not 'still' / 'moving'
    return a, X, a.groupby("run").size().values


def fit_hmm(X, lengths, k, seed=156, n_iter=300):
    best = None
    for s in range(5):                                       # EM finds local optima: keep the best of 5 starts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GaussianHMM(k, covariance_type="full", n_iter=n_iter, random_state=seed + s, min_covar=1e-2).fit(X, lengths)
            ll = m.score(X, lengths)
        if best is None or ll > best[0]:
            best = (ll, m)
    return best[1]


def forward_loglik(m, x):
    """Forward algorithm written out by hand (log space); matches hmmlearn's score for one sequence."""
    logB = np.column_stack([multivariate_normal.logpdf(x, m.means_[j], m.covars_[j]) for j in range(m.n_components)])
    logA, la = np.log(m.transmat_ + 1e-300), np.log(m.startprob_ + 1e-300) + logB[0]
    for t in range(1, len(x)):
        la = logsumexp(la[:, None] + logA, axis=0) + logB[t]
    return logsumexp(la)


def choose_k(a, ks=(1, 2, 3, 4)):
    """Leave-one-day-out log-likelihood per window for each K. Sequences are assigned to the day they start on."""
    a2, X, lengths = sequences(a)
    run_day = a2.groupby("run").date.first().values
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    rows = []
    for k in ks:
        tot, n = 0.0, 0
        for day in np.unique(run_day):
            te = run_day == day
            tr_idx = np.concatenate([np.arange(s, s + l) for s, l, t in zip(starts, lengths, te) if not t])
            tr_len = lengths[~te]
            m = fit_hmm(X[tr_idx], tr_len, k)
            for s, l in zip(starts[te], lengths[te]):
                tot += m.score(X[s:s + l]); n += l
        rows.append(dict(K=k, heldout_loglik_per_window=tot / n))
    return pd.DataFrame(rows)


def decode(a, k):
    a2, X, lengths = sequences(a)
    m = fit_hmm(X, lengths, k)
    order = np.argsort(m.means_[:, 0])                       # state 0 = lowest mean residual ("calm")
    rank = np.empty(k, int); rank[order] = np.arange(k)
    a2["state"] = rank[m.predict(X, lengths)]
    post = m.predict_proba(X, lengths)[:, order]
    a2["p_elevated"] = post[:, -1]
    return a2, m, order
