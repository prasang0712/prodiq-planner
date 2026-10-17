"""Shop calendar and the dispatching simulation used both for the forward plan and for the back-test.

Rules (the configuration that won the ProdiQ benchmark):
  queue order : Apparent Tardiness Cost (ATC) using planned (standard) times, k = 2
  machine     : conventional -> any free machine of the right type (picked at random)
                ProdiQ       -> the free machine with the lowest predicted processing time
"""
import bisect, heapq, math
from collections import defaultdict
import numpy as np
import pandas as pd

PRIORITY_WEIGHT = {"Normal": 1.0, "High": 2.0, "Urgent": 4.0}
DEFAULT_CALENDAR = {d: [(6.0, 22.0)] for d in range(5)}           # Mon-Fri 06:00-22:00


class Calendar:
    """Working windows repeated weekly. Times are minutes since t0 (a pandas Timestamp)."""
    def __init__(self, t0, weekly=None, horizon_days=1200):
        self.t0 = pd.Timestamp(t0)
        self.weekly = weekly or DEFAULT_CALENDAR
        self.week_min = sum((b - a) * 60 for ws in self.weekly.values() for a, b in ws if b > a)
        if self.week_min <= 0: raise ValueError("The shop calendar has no working hours.")
        self.f = 10080 / self.week_min                     # wall-clock minutes per working minute
        self.starts, self.ends, self.cum = [], [], [0.0]
        self._build(horizon_days)

    def _build(self, days):
        day0 = self.t0.normalize()
        for i in range(-1, days):
            d = day0 + pd.Timedelta(days=i)
            for a, b in sorted(self.weekly.get(d.weekday(), [])):
                if b <= a: continue
                s = (d + pd.Timedelta(hours=a) - self.t0).total_seconds() / 60
                e = (d + pd.Timedelta(hours=b) - self.t0).total_seconds() / 60
                if e <= 0: continue
                s = max(s, 0.0)
                self.starts.append(s); self.ends.append(e); self.cum.append(self.cum[-1] + e - s)

    def next_working(self, t):
        i = bisect.bisect_right(self.ends, t)
        if i >= len(self.starts): raise ValueError("Plan runs past the calendar horizon.")
        return max(t, self.starts[i])

    def add(self, t, work):
        """time at which `work` working minutes are done, starting at t"""
        t = self.next_working(t); i = bisect.bisect_right(self.ends, t)
        left = self.ends[i] - t
        if work <= left: return t + work
        target = self.cum[i + 1] + (work - left)                  # cumulative working minutes at the end
        j = bisect.bisect_left(self.cum, target) - 1
        if j >= len(self.starts): raise ValueError("Plan runs past the calendar horizon.")
        return self.starts[j] + (target - self.cum[j])

    def ts(self, t): return self.t0 + pd.Timedelta(minutes=float(t))
    def minutes(self, ts): return (pd.Timestamp(ts) - self.t0).total_seconds() / 60


def simulate(jobs, machines_by_type, cal, policy, planned, predicted, actual, k=2.0, transfer_min=0.0,
             machine_free_at=None, seed=0):
    """jobs: list of dict(id, route[list of types], due (min or None), release (min), weight)
    planned[i][s]           planned working minutes of job i step s (machine-independent)
    predicted[i][s][m]      ProdiQ's predicted working minutes on machine m
    actual(i, s, m) -> working minutes the operation really takes (plan: the best estimate; back-test: sampled truth)
    policy: 'conventional' or 'prodiq'
    Returns (ops list, completion time per job)."""
    rng = np.random.default_rng(seed)
    busy = {m: False for ms in machines_by_type.values() for m in ms}
    queues = defaultdict(list); ev = []; n = [0]
    idx = [0] * len(jobs); ready_t = [0.0] * len(jobs); done = [None] * len(jobs); ops = []
    def push(t, kind, d): heapq.heappush(ev, (t, n[0], kind, d)); n[0] += 1
    rem = [[sum(planned[i][s + 1:]) for s in range(len(j["route"]))] for i, j in enumerate(jobs)]

    def pick_job(q, t):
        if all(jobs[i]["due"] is None for i in q):                 # no due dates: priority, then first come
            return min(range(len(q)), key=lambda x: (-jobs[q[x]]["weight"], ready_t[q[x]]))
        ps = [max(planned[i][idx[i]], 1.0) for i in q]
        pbar = cal.f * max(np.mean(ps), 1.0); best, bi = -1.0, 0
        for x, i in enumerate(q):
            due = jobs[i]["due"] if jobs[i]["due"] is not None else t + 1e9
            slack = due - t - cal.f * ps[x] - cal.f * rem[i][idx[i]]
            v = jobs[i]["weight"] / ps[x] * math.exp(-max(0.0, slack) / (k * pbar))
            if v > best: best, bi = v, x
        return bi

    def dispatch(ty, t):
        q = queues[ty]
        while q:
            idle = [m for m in machines_by_type[ty] if not busy[m]]
            if not idle: return
            i = q.pop(pick_job(q, t)); s = idx[i]
            if policy == "prodiq":
                m = min(idle, key=lambda mm: predicted[i][s].get(mm, 1e18))
            else:
                m = idle[int(rng.integers(len(idle)))]
            st = cal.next_working(t); dur = float(actual(i, s, m)); en = cal.add(st, dur)
            busy[m] = True
            ops.append((i, s, ty, m, st, en, dur))
            push(en, "finish", (i, m))

    for i, j in enumerate(jobs): push(j["release"], "ready", i)
    for m, t_free in (machine_free_at or {}).items():
        if m in busy and t_free and t_free > 0: busy[m] = True; push(t_free, "free", m)
    mtype = {m: ty for ty, ms in machines_by_type.items() for m in ms}
    while ev:
        t, _, kind, d = heapq.heappop(ev)
        if kind == "ready":
            i = d; ready_t[i] = t; queues[jobs[i]["route"][idx[i]]].append(i); dispatch(jobs[i]["route"][idx[i]], t)
        elif kind == "finish":
            i, m = d; busy[m] = False; idx[i] += 1
            if idx[i] < len(jobs[i]["route"]): push(t + transfer_min, "ready", i)
            else: done[i] = t
            dispatch(mtype[m], t)
        elif kind == "free":
            busy[d] = False; dispatch(mtype[d], t)
    return ops, done


def summarise(jobs, done, cal):
    late = [done[i] > j["due"] for i, j in enumerate(jobs) if j["due"] is not None and j.get("score", True)]
    tard = [max(0.0, done[i] - j["due"]) / 60 for i, j in enumerate(jobs) if j["due"] is not None and j.get("score", True)]
    lead = [(done[i] - j["release"]) / 60 for i, j in enumerate(jobs) if j.get("score", True)]
    urg = [done[i] > j["due"] for i, j in enumerate(jobs)
           if j["due"] is not None and j.get("score", True) and j.get("priority") == "Urgent"]
    out = dict(orders=len(lead), mean_lead_h=float(np.mean(lead)), p95_lead_h=float(np.percentile(lead, 95)))
    if late:
        out.update(on_time=1 - float(np.mean(late)), late_orders=int(np.sum(late)), mean_tardiness_h=float(np.mean(tard)))
    if urg: out["urgent_on_time"] = 1 - float(np.mean(urg))
    return out
