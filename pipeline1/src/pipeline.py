"""End-to-end run: watch CSVs + calendar tags -> readiness, calm/elevated states, weekend advice."""
import os
import numpy as np
import pandas as pd
from features import build_windows
from daily import build_daily
from hmm_states import decode
from advice import playbook, week_summary, weekend_plan


def last_work_week(d):
    wd = d.index[d.index.dayofweek < 5]
    return wd[-5:]


def run_all(watch_dir, calendar_path, out_dir, k_states=2):
    os.makedirs(out_dir, exist_ok=True)
    w, _ = build_windows(watch_dir, calendar_path)
    d, a, info, mor = build_daily(w, watch_dir)
    a2, hmm, order = decode(a, k_states)
    elevated = a2.groupby("date").state.mean()
    pb = playbook(d)
    wk = last_work_week(d)
    summary = week_summary(d, mor, elevated, wk)
    plan = weekend_plan(summary, pb)
    mor[["asleep_h", "composite", "score", "score_lo", "score_hi"]].round(2).to_csv(f"{out_dir}/readiness.csv")
    lines = [f"# Weekend plan (from {wk[0]:%b %d} to {wk[-1]:%b %d})", ""]
    lines += [f"- **{p['area']}**: {p['advice']}  \n  *Why:* {p['because']} *[{p['source']}]*" for p in plan]
    lines += ["", "_General wellbeing suggestions from your own watch data. Not medical advice._"]
    open(f"{out_dir}/weekend_plan.md", "w").write("\n".join(lines))
    return dict(daily=d, windows=a2, mornings=mor, plan=plan, summary=summary, playbook=pb, info=info)
