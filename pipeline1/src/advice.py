"""Weekend recovery advice from the week that just ended.

Three layers, always labelled so the reader knows how much to trust each line:
  1. MY DATA        what the week looked like compared with my own typical week (percentiles / SD of my other days)
  2. PLAYBOOK       what tended to come before better mornings for me (shrunk toward zero because there are so few days)
  3. GENERAL        standard recovery guidance, triggered by layer 1, plus my own stated preference
                    (calming rest rather than sport when strained; dance counts as good strain when I am recovered)
Every rule and threshold sits in RULES so anyone can change it.
"""
import numpy as np
import pandas as pd

RULES = dict(
    sleep_debt_h=3.0,         # total hours below my usual sleep over the week that counts as "debt"
    short_night_h=6.0,
    readiness_low=-0.5,       # composite z below this = a rough week
    high_strain_S=1.5,
    costly_h_hi=10,           # hours of class, meetings, work, hard tasks and travel in a week that count as a costly week
    elevated_share_hi=0.10,   # extra share of awake time in the elevated state, vs my usual
    low_steps_ratio=0.75,     # steps below 75 % of my median day = sedentary week
    low_daylight_min=30,
    loud_db=65,
    late_bed_h=2.5,           # falling asleep after 02:30
    min_days_for_claim=4,     # an activity seen on fewer days than this never gets a 'clear' effect
    calm_not_sport=True,      # personal preference: when strained, recover with calm, not exercise
    shrink_n0=3,              # prior strength: a tag seen on n days is trusted n/(n+n0) as much as the raw difference
)
PLAYBOOK_TAGS = ["walk", "exercise", "nap", "social", "travel", "study", "class_cs156", "class_other", "work"]


def playbook(d, target="m_composite", boot=500, seed=156):
    """For each activity: next-morning readiness on days with >= 1 h of it vs days without, shrunk toward 0."""
    rng = np.random.default_rng(seed)
    x = d.dropna(subset=[target]).copy()
    x["h_social"] = x.h_social + x.h_social_late
    rows = []
    for t in PLAYBOOK_TAGS:
        on = x[f"h_{t}"] >= 1.0
        n_on = int(on.sum())
        if n_on < 1 or n_on == len(x):
            continue
        diff = x.loc[on, target].mean() - x.loc[~on, target].mean()
        shrunk = diff * n_on / (n_on + RULES["shrink_n0"])
        bs = []
        for _ in range(boot):
            s = x.iloc[rng.integers(0, len(x), len(x))]
            o = s[f"h_{t}"] >= 1.0
            if o.sum() and (~o).sum():
                bs.append((s.loc[o, target].mean() - s.loc[~o, target].mean()) * o.sum() / (o.sum() + RULES["shrink_n0"]))
        lo, hi = np.percentile(bs, [5, 95])
        rows.append(dict(activity=t, days_with=n_on, raw_diff=diff, shrunk_diff=shrunk, lo=lo, hi=hi, clear=((lo > 0) or (hi < 0)) and n_on >= RULES["min_days_for_claim"]))
    return pd.DataFrame(rows).sort_values("shrunk_diff", ascending=False).reset_index(drop=True)


def analog_days(d, query_days, before, k=3, cols=("S", "steps_k", "daylight_min", "noise_db", "cognitive_h", "this_composite")):
    """k nearest past days (strictly before `before`) to the average of the query days, and what followed them."""
    x = d.copy(); x["steps_k"] = x.steps / 1000
    past = x[x.index < before].dropna(subset=list(cols) + ["m_composite"])
    q = x.loc[x.index.isin(query_days), list(cols)].mean()
    mu, sd = past[list(cols)].mean(), past[list(cols)].std(ddof=0).replace(0, 1)
    dist = (((past[list(cols)] - q) / sd) ** 2).sum(axis=1) ** 0.5
    nn = dist.sort_values().index[:k]
    return pd.DataFrame({"distance": dist[nn], "next_morning_composite": x.loc[nn, "m_composite"]})


def week_summary(d, mor, elevated_share, week_days):
    """Compare a week with my own baseline. `d` is the daily table, `mor` the morning table, `elevated_share` per day."""
    wk = d.loc[d.index.isin(week_days)]
    base_sleep = mor.asleep_h.mean()
    mm = mor.loc[mor.index.isin(week_days)]
    return dict(
        sleep_debt_h=float(np.clip(base_sleep - mm.asleep_h, 0, None).sum()), base_sleep_h=float(base_sleep),
        short_nights=int((mm.asleep_h < RULES["short_night_h"]).sum()),
        late_beds=int((mm.bedtime_h > RULES["late_bed_h"]).sum()),
        mean_readiness=float(mm.composite.mean()), last_readiness=float(mm.composite.dropna().iloc[-1]) if mm.composite.notna().any() else np.nan,
        high_strain_days=int((wk.S > RULES["high_strain_S"]).sum()),
        costly_h=float(wk.costly_h.sum()), refreshing_h=float(wk.refreshing_h.sum()),
        elevated_extra=float(elevated_share.reindex(wk.index).mean() - elevated_share.mean()),
        steps_ratio=float(wk.steps.mean() / d.steps.median()), daylight=float(wk.daylight_min.mean()),
        noise=float(wk.noise_db.mean()), travel_h=float(wk.h_travel.sum()), cognitive_h=float(wk.cognitive_h.sum()),
        unwell=bool(wk.unwell.any()),
    )


def weekend_plan(s, pb):
    """Rule layer: returns a list of dicts(area, advice, because, source). Nothing here is medical advice."""
    R, plan = RULES, []
    add = lambda area, advice, because, source: plan.append(dict(area=area, advice=advice, because=because, source=source))
    rough = s["mean_readiness"] < R["readiness_low"] or s["unwell"]
    if s["sleep_debt_h"] >= R["sleep_debt_h"] or s["short_nights"] >= 1:
        extra = min(1.5, s["sleep_debt_h"] / 3)
        add("Sleep", f"Aim for about {s['base_sleep_h'] + extra:.1f} h on both nights and keep the wake-up within an hour of the weekday time.",
            f"{s['short_nights']} short night(s) and {s['sleep_debt_h']:.1f} h of sleep debt this week (my usual is {s['base_sleep_h']:.1f} h).",
            "my data + general guidance")
    if s["late_beds"] >= 2:
        add("Sleep", "Pick one fixed bedtime for Saturday and hold it. Late bedtimes made up most of this week.",
            f"{s['late_beds']} nights started after 02:30.", "my data + general guidance")
    costly_week = s["costly_h"] >= R["costly_h_hi"]
    if rough or costly_week:
        add("Calm", "Two or three 15 to 20 minute calm blocks a day: slow breathing (a Mindfulness session on the watch logs it), a bath or quiet reading, no screens.",
            (f"{s['costly_h']:.0f} h of class, meetings, work, hard tasks and travel this week (costly week from {R['costly_h_hi']} h)" if costly_week else "a rough week (low readiness or illness)") + f"; elevated-state time was {100*s['elevated_extra']:+.0f} points vs my usual.",
            "my data + general guidance")
        add("Refresh", "Plan one low-key thing I find refreshing, like time with friends or a slow walk. Dance only if I feel recovered and am not ill.",
            f"I find friends, dancing and walking refreshing, and class, work and travel costly; this week had {s['costly_h']:.0f} h costly vs {s['refreshing_h']:.0f} h refreshing.", "my stated preference")
        add("Movement", "No sport this weekend. Keep movement easy: a slow walk, stretching." if R["calm_not_sport"] else "Keep exercise light and optional.", "My setting: strained weeks call for calming rest, not sport (RULES['calm_not_sport']).", "my stated preference")
    else:
        add("Movement", "I am recovered enough: dance or a longer walk is fine. It counts as good strain for me.",
            f"Readiness averaged {s['mean_readiness']:+.1f} SD and strain stayed low.", "my stated preference")
    if s["steps_ratio"] < R["low_steps_ratio"] or s["cognitive_h"] >= 15:
        add("Daylight", "Get 30 to 60 minutes outside in daylight, ideally in the morning, at an easy pace.",
            f"Desk-heavy week ({s['cognitive_h']:.0f} h of class, study and work; steps at {100*s['steps_ratio']:.0f}% of my median day).", "my data + general guidance")
    elif s["daylight"] < R["low_daylight_min"]:
        add("Daylight", "Plan one daylight walk. My daylight time was low.", f"{s['daylight']:.0f} min a day on average.", "my data + general guidance")
    if s["noise"] >= R["loud_db"] or s["travel_h"] >= 6:
        add("Quiet", "Keep one day low-input: few plans, little noise, no transit.", f"Travel {s['travel_h']:.0f} h and average noise {s['noise']:.0f} dB this week.", "my data + general guidance")
    clear = pb[pb.clear & (pb.shrunk_diff > 0)] if len(pb) else pb
    if len(clear):
        t = clear.iloc[0]
        add("Playbook", f"{t.activity} has gone with better next mornings for me.", f"{t.days_with} days, {t.shrunk_diff:+.2f} SD (interval {t.lo:+.2f} to {t.hi:+.2f}).", "my data")
    else:
        add("Playbook", "Not enough days yet to say which activities help me most. Keep logging labels and mindful sessions.",
            f"No activity has been seen on {RULES['min_days_for_claim']}+ days with an interval that excludes zero.", "my data")
    return plan
