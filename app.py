"""ProdiQ web app - machine-aware production planning for any job shop.
Run locally:  pip install -r requirements.txt   then   streamlit run app.py
"""
import hashlib, io, os
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================ ProdiQ engine (inlined)
import bisect, heapq, math
from collections import defaultdict
from dataclasses import dataclass, field
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
# ---------------- engine: data.py
ALIASES = {
    "machine_type": ["machine_type", "op_type", "operation_type", "operation", "work_center", "workcenter",
                     "work_centre", "workcentre", "resource_type", "process", "station_type", "type"],
    "machine_id": ["machine_id", "machine", "resource", "resource_id", "station", "station_id", "equipment",
                   "equipment_id", "asset", "asset_id"],
    "order_id": ["order_id", "wo_id", "work_order", "work_order_id", "job", "job_id", "order", "order_no",
                 "order_number", "job_no", "job_number", "wo", "wo_no", "mo", "mo_id", "case", "case_id"],
    "part": ["part", "part_no", "part_number", "product", "product_id", "item", "item_code", "sku", "material"],
    "quantity": ["quantity", "qty", "qty_in", "batch_size", "lot_size", "units", "order_qty"],
    "run_minutes": ["run_minutes", "actual_time_min", "actual_minutes", "duration_min", "duration_minutes",
                    "processing_time_min", "proc_time_min", "run_time_min", "minutes"],
    "planned_minutes": ["planned_minutes", "std_time_min", "standard_time_min", "standard_minutes", "planned_time_min",
                        "routing_time_min", "std_minutes"],
    "start_time": ["start_time", "start_ts", "start", "started_at", "start_date"],
    "end_time": ["end_time", "end_ts", "end", "finished_at", "end_date", "complete_time"],
    "due_date": ["due_date", "due_ts", "due", "due_time", "promise_date", "delivery_date"],
    "priority": ["priority", "prio", "urgency"],
    "sequence": ["sequence", "seq", "step", "op", "op_no", "op_seq", "operation_no", "operation_number"],
    "route": ["route", "routing", "operations", "remaining_route", "process_route"],
    "release_time": ["release_time", "release_ts", "release", "release_date", "order_date", "created_at"],
    "available_from": ["available_from", "busy_until", "free_from"],
}
ROUTE_SEPARATORS = [">", "->", "→", ",", ";", "|"]


@dataclass
class CheckResult:
    table: pd.DataFrame = None
    errors: list = field(default_factory=list)      # stop: the file cannot be used
    warnings: list = field(default_factory=list)    # usable, but something to know
    notes: list = field(default_factory=list)       # what was done automatically

    @property
    def ok(self): return not self.errors


def _norm(c):
    return str(c).strip().lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def standardise_columns(df, wanted):
    """rename columns to the standard names using ALIASES; returns (df, mapping used)"""
    cols = {_norm(c): c for c in df.columns}
    ren, used = {}, {}
    for std in wanted:
        for a in ALIASES[std]:
            if a in cols and cols[a] not in ren:
                ren[cols[a]] = std; used[std] = cols[a]; break
    return df.rename(columns=ren), used


def _read(f):
    if isinstance(f, pd.DataFrame): return f.copy()
    name = getattr(f, "name", str(f)).lower()
    if name.endswith((".xlsx", ".xls")): return pd.read_excel(f)
    return pd.read_csv(f, sep=None, engine="python")      # sniff comma / semicolon / tab


def _mapping_note(used, table):
    renamed = [f"'{v}' as {k}" for k, v in used.items() if _norm(v) != k]
    return [f"{table}: read column " + ", ".join(renamed)] if renamed else []


def check_machines(f):
    r = CheckResult()
    try: df = _read(f)
    except Exception as e:
        r.errors.append(f"Machines file could not be read ({e})."); return r
    df, used = standardise_columns(df, ["machine_id", "machine_type", "available_from"])
    r.notes += _mapping_note(used, "Machines")
    miss = [c for c in ("machine_id", "machine_type") if c not in df]
    if miss:
        r.errors.append(f"Machines file needs the column(s) {', '.join(miss)}. Found: {', '.join(map(str, df.columns))}."); return r
    df["machine_id"] = df.machine_id.astype(str).str.strip(); df["machine_type"] = df.machine_type.astype(str).str.strip()
    dup = df.machine_id[df.machine_id.duplicated()].unique()
    if len(dup): r.warnings.append(f"Machines listed twice (kept the first row): {', '.join(dup[:10])}.")
    df = df.drop_duplicates("machine_id")
    if "available_from" in df:
        df["available_from"] = pd.to_datetime(df.available_from, errors="coerce")
    r.table = df.reset_index(drop=True)
    return r


def check_history(f, machines=None):
    r = CheckResult()
    try: df = _read(f)
    except Exception as e:
        r.errors.append(f"History file could not be read ({e})."); return r
    df, used = standardise_columns(df, ["order_id", "machine_type", "machine_id", "part", "quantity", "run_minutes",
                                        "planned_minutes", "start_time", "end_time", "due_date", "priority", "sequence"])
    r.notes += _mapping_note(used, "History")
    need = ["order_id", "machine_type", "machine_id", "part", "quantity"]
    miss = [c for c in need if c not in df]
    if miss:
        r.errors.append(f"History file needs the column(s) {', '.join(miss)}. Found: {', '.join(map(str, df.columns))}."); return r
    for c in ("start_time", "end_time", "due_date"):
        if c in df: df[c] = pd.to_datetime(df[c], errors="coerce")
    if "run_minutes" not in df:
        if "start_time" in df and "end_time" in df:
            df["run_minutes"] = (df.end_time - df.start_time).dt.total_seconds() / 60
            r.warnings.append("No run-time column, so run time = end time − start time. If operations span nights or "
                              "weekends this overstates run time; add a run_minutes column if you have one.")
        else:
            r.errors.append("History file needs run_minutes, or both start_time and end_time."); return r
    for c in ("order_id", "machine_type", "machine_id", "part"): df[c] = df[c].astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df.quantity, errors="coerce")
    df["run_minutes"] = pd.to_numeric(df.run_minutes, errors="coerce")
    if "planned_minutes" in df: df["planned_minutes"] = pd.to_numeric(df.planned_minutes, errors="coerce")
    bad = df.quantity.isna() | (df.quantity <= 0) | df.run_minutes.isna() | (df.run_minutes <= 0)
    if bad.any():
        r.warnings.append(f"Skipped {int(bad.sum())} history rows with a missing or zero quantity or run time.")
        df = df[~bad]
    if "planned_minutes" in df:
        share = df.planned_minutes.notna().mean()
        if share < 0.8:
            r.notes.append(f"Planned times are filled in for only {share:.0%} of rows, so ProdiQ estimates its own "
                           "standard times from history instead.")
            df = df.drop(columns="planned_minutes")
    if machines is not None:
        known = set(machines.machine_id)
        unk = sorted(set(df.machine_id) - known)
        if unk:
            r.warnings.append(f"{len(unk)} machine(s) in the history are not in the machines file and were ignored: "
                              f"{', '.join(unk[:8])}{'…' if len(unk) > 8 else ''}.")
            df = df[df.machine_id.isin(known)]
        mt = machines.set_index("machine_id").machine_type
        mism = df.machine_type != df.machine_id.map(mt)
        if mism.any():
            r.notes.append(f"{int(mism.sum())} history rows had a machine type different from the machines file; "
                           "the machines file was used.")
            df.loc[mism, "machine_type"] = df.loc[mism, "machine_id"].map(mt)
    # operation order inside each order
    if "start_time" in df and df.start_time.notna().mean() > 0.95:
        df = df.sort_values(["start_time"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        r.warnings.append("No usable start times: the history is taken in file order, and the 'Will it help here?' "
                          "test cannot run (it needs to know when orders arrived).")
    if "sequence" in df:
        df["sequence"] = pd.to_numeric(df.sequence, errors="coerce")
    df["row"] = np.arange(len(df))
    if len(df) < 100:
        r.errors.append(f"Only {len(df)} usable history rows. ProdiQ needs at least 100 finished operations to learn from.")
    r.table = df
    return r


def split_route(s):
    s = str(s)
    for sep in ROUTE_SEPARATORS:
        if sep in s: return [x.strip() for x in s.split(sep) if x.strip()]
    return [s.strip()] if s.strip() else []


def check_orders(f, machines=None, default_release=None):
    r = CheckResult()
    try: df = _read(f)
    except Exception as e:
        r.errors.append(f"Open-orders file could not be read ({e})."); return r
    df, used = standardise_columns(df, ["order_id", "part", "quantity", "due_date", "route", "priority", "release_time"])
    r.notes += _mapping_note(used, "Orders")
    miss = [c for c in ("order_id", "part", "quantity", "due_date", "route") if c not in df]
    if miss:
        r.errors.append(f"Open-orders file needs the column(s) {', '.join(miss)}. Found: {', '.join(map(str, df.columns))}."); return r
    df["order_id"] = df.order_id.astype(str).str.strip(); df["part"] = df.part.astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df.quantity, errors="coerce")
    df["due_date"] = pd.to_datetime(df.due_date, errors="coerce")
    df["route_list"] = df.route.map(split_route)
    if "priority" not in df: df["priority"] = "Normal"
    df["priority"] = df.priority.fillna("Normal").astype(str).str.strip().str.capitalize()
    oddp = sorted(set(df.priority) - {"Normal", "High", "Urgent"})
    if oddp:
        r.notes.append(f"Unknown priority values {', '.join(oddp)} were treated as Normal.")
        df.loc[~df.priority.isin(["Normal", "High", "Urgent"]), "priority"] = "Normal"
    if "release_time" in df: df["release_time"] = pd.to_datetime(df.release_time, errors="coerce")
    bad = df.quantity.isna() | (df.quantity <= 0) | df.due_date.isna() | (df.route_list.str.len() == 0)
    if bad.any():
        r.warnings.append(f"Skipped {int(bad.sum())} orders with a missing quantity, due date or route: "
                          f"{', '.join(df.order_id[bad].head(8))}.")
        df = df[~bad]
    dup = df.order_id[df.order_id.duplicated()].unique()
    if len(dup):
        r.warnings.append(f"Order IDs listed twice (kept the first): {', '.join(dup[:8])}."); df = df.drop_duplicates("order_id")
    if machines is not None:
        types = set(machines.machine_type)
        unk = sorted({t for rt in df.route_list for t in rt} - types)
        if unk:
            r.errors.append(f"Routes use machine types that are not in the machines file: {', '.join(unk)}. "
                            f"Known types: {', '.join(sorted(types))}.")
    if df.empty and not r.errors: r.errors.append("No usable open orders.")
    r.table = df.reset_index(drop=True)
    return r


def parse_weekly_calendar(days, start_h, end_h):
    """days: list of weekday numbers (Mon=0). Same working window every working day."""
    return {d: [(float(start_h), float(end_h))] for d in days}

# ---------------- engine: plan.py
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

# ---------------- engine: model.py
CAT = ["machine_type", "machine_id", "part"]
NUM = ["log_qty", "log_std"]


def _shrunk(groups, k=10.0):
    return {g: float(s.sum() / (len(s) + k)) for g, s in groups}


class StandardTimes:
    def fit(self, h):
        self.use_planned = "planned_minutes" in h
        lw, lq = np.log(h.run_minutes), np.log(h.quantity)
        self.mean_lq = float(lq.mean())
        self.mu = lw.groupby(h.machine_type).mean().to_dict()
        r0 = lw - h.machine_type.map(self.mu)
        self.part = _shrunk(r0.groupby(h.part))
        r1 = r0 - h.part.map(self.part).fillna(0.0)
        x = lq - self.mean_lq
        self.b = float(np.clip((x * r1).sum() / max((x ** 2).sum(), 1e-9), 0.0, 1.0))
        self.global_mu = float(lw.mean())
        if self.use_planned:      # per (type, part): planned minutes per unit, for open orders
            pu = (h.planned_minutes / h.quantity)
            self.pp_unit = pu.groupby([h.machine_type, h.part]).median().to_dict()
            self.pt_unit = pu.groupby(h.machine_type).median().to_dict()
        return self

    def estimate(self, mtype, part, qty):
        if self.use_planned:
            u = self.pp_unit.get((mtype, part), self.pt_unit.get(mtype))
            if u is not None: return float(u * qty)
        return float(np.exp(self.mu.get(mtype, self.global_mu) + self.part.get(part, 0.0)
                            + self.b * (np.log(qty) - self.mean_lq)))

    def for_rows(self, df):
        if self.use_planned and "planned_minutes" in df:
            est = df.planned_minutes
            fill = [self.estimate(t, p, q) for t, p, q in zip(df.machine_type, df.part, df.quantity)]
            return est.fillna(pd.Series(fill, index=df.index)).to_numpy(float)
        return np.array([self.estimate(t, p, q) for t, p, q in zip(df.machine_type, df.part, df.quantity)])


def _features(df, std):
    return pd.DataFrame(dict(machine_type=df.machine_type.values, machine_id=df.machine_id.values, part=df.part.values,
                             log_qty=np.log(df.quantity.values), log_std=np.log(np.maximum(std, 0.1))))


class TimeModel:
    def __init__(self, n_trees=150, seed=42):
        self.n_trees, self.seed = n_trees, seed

    def fit(self, h, std_model):
        self.std = std_model
        s = std_model.for_rows(h)
        X = _features(h, s); y = np.log1p(h.run_minutes.to_numpy(float))
        leaf = int(np.clip(len(h) // 400, 3, 20))
        self.pipe = Pipeline([("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), CAT),
                                                         ("num", "passthrough", NUM)])),
                              ("rf", RandomForestRegressor(n_estimators=self.n_trees, max_depth=20, min_samples_leaf=leaf,
                                                           max_features=0.5, random_state=self.seed, n_jobs=1))])
        self.pipe.fit(X, y)
        pred = np.expm1(self.pipe.predict(X))
        g = pd.DataFrame(dict(t=h.machine_type.values, a=h.run_minutes.values, p=pred))
        self.smear = (g.groupby("t").a.sum() / g.groupby("t").p.sum()).to_dict()
        return self

    def predict(self, df, std=None, mean=True):
        """mean=True: expected minutes (for adding up a schedule); False: typical (median) minutes"""
        std = self.std.for_rows(df) if std is None else std
        p = np.expm1(self.pipe.predict(_features(df, std)))
        return p * df.machine_type.map(self.smear).fillna(1.0).to_numpy(float) if mean else p

    def predict_all_machines(self, rows, machines):
        """rows: DataFrame (machine_type, part, quantity [, planned_minutes]) -> DataFrame with one column per machine
        of the row's type (NaN elsewhere); index aligned with rows."""
        rows = rows.reset_index(drop=True)
        std = self.std.for_rows(rows)
        by_type = machines.groupby("machine_type").machine_id.apply(list).to_dict()
        parts = []
        for ty, ms in by_type.items():
            sel = np.where(rows.machine_type.values == ty)[0]
            if not len(sel): continue
            for m in ms:
                r = rows.iloc[sel].assign(machine_id=m)
                parts.append(pd.DataFrame(dict(row=sel, machine_id=m, pred=self.predict(r, std[sel]))))
        out = pd.concat(parts).pivot(index="row", columns="machine_id", values="pred") if parts else pd.DataFrame()
        return out.reindex(range(len(rows))), std


def accuracy(h_test, model, std_model):
    """compare the conventional standard times and ProdiQ on held-out operations"""
    a = h_test.run_minutes.to_numpy(float)
    s = std_model.for_rows(h_test); p = model.predict(h_test, s, mean=False)
    def m(x): return dict(mae=float(np.mean(np.abs(a - x))), wape=float(np.sum(np.abs(a - x)) / np.sum(a)))
    return {"Standard times (conventional)": m(s), "ProdiQ prediction": m(p)}


def machine_speeds(h, model, machines):
    """how much faster/slower each machine is than the average machine of its type, according to the model,
    over the jobs that type actually ran; plus the observed figure (actual vs standard time) for comparison"""
    rows = h[["machine_type", "part", "quantity"] + (["planned_minutes"] if "planned_minutes" in h else [])]
    P, std = model.predict_all_machines(rows, machines)
    out = []
    obs = np.log(h.run_minutes.to_numpy(float) / std)
    for ty, ms in machines.groupby("machine_type").machine_id.apply(list).items():
        sel = (h.machine_type == ty).to_numpy()
        if not sel.any(): continue
        lp = np.log(P.loc[sel, ms].to_numpy(float))
        rel = lp - lp.mean(axis=1, keepdims=True)                      # vs type average, same jobs
        ob = pd.Series(obs[sel]).groupby(h.machine_id.to_numpy()[sel]).agg(["mean", "size"])
        ob["mean"] -= obs[sel].mean()                                  # observed, vs type average
        for k, m in enumerate(ms):
            out.append(dict(machine_type=ty, machine_id=m, n_ops=int(ob["size"].get(m, 0)),
                            predicted_vs_type_avg=float(np.exp(rel[:, k].mean()) - 1),
                            observed_vs_type_avg=float(np.exp(ob["mean"].get(m, np.nan)) - 1) if m in ob.index else np.nan))
    return pd.DataFrame(out)


def ranking_agreement(h_test, model, machines):
    """share of machine pairs (same type, both with >= 10 held-out operations) that the model orders the same way as
    what really happened (actual time relative to standard time)"""
    rows = h_test[["machine_type", "part", "quantity"] + (["planned_minutes"] if "planned_minutes" in h_test else [])]
    P, std = model.predict_all_machines(rows, machines)
    obs = pd.Series(np.log(h_test.run_minutes.to_numpy(float) / std)).groupby(h_test.machine_id.to_numpy())
    cnt, mean_obs = obs.size(), obs.mean()
    agree = total = 0
    for ty, ms in machines.groupby("machine_type").machine_id.apply(list).items():
        ok = [m for m in ms if cnt.get(m, 0) >= 10]
        sel = (h_test.machine_type == ty).to_numpy()
        if len(ok) < 2 or not sel.any(): continue
        pm = {m: float(np.log(P.loc[sel, m]).mean() - np.log(std[sel]).mean()) for m in ok}
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                x, y = ok[a], ok[b]
                total += 1; agree += (pm[x] < pm[y]) == (mean_obs[x] < mean_obs[y])
    return (agree / total, total) if total else None

# ---------------- engine: core.py
def _orders_from_history(h):
    """rebuild orders (route, quantity, release, due) from finished operations"""
    key = ["order_id", "sequence", "start_time"] if "sequence" in h else ["order_id", "start_time"]
    h = h.sort_values(key)
    agg = dict(part=("part", "first"), quantity=("quantity", "first"), release=("start_time", "min"),
               n_ops=("row", "size"))
    if "due_date" in h: agg["due"] = ("due_date", "first")
    if "priority" in h: agg["priority"] = ("priority", "first")
    o = h.groupby("order_id", sort=False).agg(**agg)
    o["rows"] = h.groupby("order_id", sort=False).row.apply(list)
    return o.sort_values("release")


class Factory:
    def __init__(self, machines, history, weekly_calendar, transfer_min=0.0):
        self.machines, self.h, self.weekly, self.transfer = machines, history, weekly_calendar, transfer_min
        self.by_type = machines.groupby("machine_type").machine_id.apply(list).to_dict()

    # ------------------------------------------------------------------ learning
    def learn(self, test_share=0.2):
        h = self.h
        n_test = int(len(h) * test_share)
        tr, te = h.iloc[: len(h) - n_test], h.iloc[len(h) - n_test:]
        std_tr = StandardTimes().fit(tr)
        m_tr = TimeModel().fit(tr, std_tr)
        self.accuracy = accuracy(te, m_tr, std_tr) if len(te) >= 30 else None
        self.agreement = ranking_agreement(te, m_tr, self.machines) if len(te) >= 30 else None
        # final models on all history
        self.std = StandardTimes().fit(h)
        self.model = TimeModel().fit(h, self.std)
        self.speeds = machine_speeds(h, self.model, self.machines)
        sp = self.speeds[self.speeds.machine_type.map(lambda t: len(self.by_type[t]) > 1)]
        self.spread = (sp.groupby("machine_type").predicted_vs_type_avg.agg(lambda x: x.max() - x.min())
                       if len(sp) else pd.Series(dtype=float))
        x = np.log1p(sp.predicted_vs_type_avg.astype(float))
        self.spread_rms = float(np.sqrt(((x - x.groupby(sp.machine_type).transform("mean")) ** 2).mean())) if len(sp) else 0.0
        return self

    # ------------------------------------------------------------------ forward plan
    def plan(self, orders, start, conventional_runs=20):
        cal = Calendar(start, self.weekly)
        jobs, steps = [], []
        for i, o in enumerate(orders.itertuples()):
            rel = 0.0
            if hasattr(o, "release_time") and pd.notna(getattr(o, "release_time", None)):
                rel = max(0.0, cal.minutes(o.release_time))
            jobs.append(dict(id=o.order_id, route=list(o.route_list), due=cal.minutes(o.due_date), release=rel,
                             weight=PRIORITY_WEIGHT.get(o.priority, 1.0), priority=o.priority))
            for s, ty in enumerate(o.route_list):
                steps.append(dict(job=i, step=s, machine_type=ty, part=o.part, quantity=float(o.quantity)))
        st = pd.DataFrame(steps)
        P, std = self.model.predict_all_machines(st[["machine_type", "part", "quantity"]], self.machines)
        planned = [[0.0] * len(j["route"]) for j in jobs]; pred = [[None] * len(j["route"]) for j in jobs]
        for r, row in enumerate(st.itertuples()):
            planned[row.job][row.step] = float(std[r])
            pred[row.job][row.step] = {m: float(P.at[r, m]) for m in self.by_type[row.machine_type]}
        free = {}
        if "available_from" in self.machines:
            for m, t in zip(self.machines.machine_id, self.machines.available_from):
                if pd.notna(t): free[m] = cal.minutes(t)
        best = lambda i, s, m: pred[i][s][m]        # best estimate of how long it takes on that machine
        ops, done = simulate(jobs, self.by_type, cal, "prodiq", planned, pred, best, transfer_min=self.transfer,
                             machine_free_at=free)
        conv = []
        for seed in range(conventional_runs):
            o_c, d_c = simulate(jobs, self.by_type, cal, "conventional", planned, pred, best,
                                transfer_min=self.transfer, machine_free_at=free, seed=seed)
            conv.append((summarise(jobs, d_c, cal), d_c))
        gantt = pd.DataFrame([dict(order_id=jobs[i]["id"], step=s + 1, machine_type=ty, machine_id=m,
                                   start=cal.ts(a).round("min"), end=cal.ts(b).round("min"), minutes=round(d, 1),
                                   standard_minutes=round(planned[i][s], 1)) for i, s, ty, m, a, b, d in ops])
        cfin = np.array([[d[i] for i in range(len(jobs))] for _, d in conv])
        od = orders[["order_id", "part", "quantity", "priority", "due_date"]].copy()
        od["route"] = orders.route_list.map(" > ".join)
        od["finish_prodiq"] = [cal.ts(d).round("min") for d in done]
        od["late_prodiq"] = [done[i] > j["due"] for i, j in enumerate(jobs)]
        od["hours_early_late"] = [round((done[i] - j["due"]) / 60, 1) for i, j in enumerate(jobs)]
        od["late_conventional_chance"] = (cfin > np.array([j["due"] for j in jobs])).mean(axis=0)
        kp = summarise(jobs, done, cal)
        kc = {k: float(np.mean([c[0][k] for c in conv])) for k in conv[0][0]}
        util = gantt.assign(h=gantt.minutes / 60).groupby("machine_id").h.sum()
        return dict(gantt=gantt, orders=od, kpi_prodiq=kp, kpi_conventional=kc, start=cal.t0,
                    end=gantt.end.max() if len(gantt) else cal.t0, machine_hours=util)

    # ------------------------------------------------------------------ will it help here?
    def backtest(self, test_share=0.25, warmup_share=0.15, reps=10, progress=None):
        h = self.h
        if "start_time" not in h or h.start_time.isna().mean() > 0.05:
            return dict(error="The history needs start times for this test.")
        o = _orders_from_history(h)
        n = len(o)
        if n < 60: return dict(error=f"Only {n} orders in the history; this test needs at least 60.")
        i_test = int(n * (1 - test_share)); i_warm = int(n * (1 - test_share - warmup_share))
        t_test = o.release.iloc[i_test]
        tr_rows = h.start_time < t_test
        std = StandardTimes().fit(h[tr_rows]); mdl = TimeModel(n_trees=100).fit(h[tr_rows], std)
        sim_o = o.iloc[i_warm:]
        cal = Calendar(sim_o.release.min().normalize(), self.weekly)
        has_due = "due" in sim_o and sim_o.due.notna().mean() > 0.9
        rows = h.loc[[r for rr in sim_o.rows for r in rr]]
        P, stdv = mdl.predict_all_machines(rows[["machine_type", "part", "quantity"] +
                                                (["planned_minutes"] if "planned_minutes" in rows else [])], self.machines)
        # what really happens: each machine's own record of actual vs standard time (test period where possible)
        all_std = std.for_rows(h); lr = pd.Series(np.log(h.run_minutes.to_numpy(float) / all_std), index=h.index)
        test_rows = h.start_time >= t_test
        pools, tpools = {}, {}
        for ty, ms in self.by_type.items():
            tp = lr[(h.machine_type == ty).to_numpy()]
            tpools[ty] = np.sort(tp.to_numpy()) if len(tp) else np.array([0.0])
            for m in ms:
                x = lr[((h.machine_id == m) & test_rows).to_numpy()]
                if len(x) < 8: x = lr[(h.machine_id == m).to_numpy()]
                pools[m] = np.sort(x.to_numpy()) if len(x) >= 8 else tpools[ty]
        jobs, planned, pred = [], [], []
        r = 0
        for k, (oid, x) in enumerate(sim_o.iterrows()):
            route = list(h.loc[x.rows, "machine_type"])
            jobs.append(dict(id=oid, route=route, release=cal.minutes(x.release),
                             due=cal.minutes(x.due) if has_due and pd.notna(x.due) else None,
                             weight=PRIORITY_WEIGHT.get(str(x.get("priority", "Normal")).capitalize(), 1.0),
                             priority=str(x.get("priority", "Normal")).capitalize(), score=k >= i_test - i_warm))
            planned.append([float(v) for v in stdv[r:r + len(route)]])
            pred.append([{m: float(P.at[r + s, m]) for m in self.by_type[route[s]]} for s in range(len(route))])
            r += len(route)
        res = {"conventional": [], "prodiq": []}
        for rep in range(reps):
            u = np.random.default_rng([rep, 7]).random((len(jobs), max(len(j["route"]) for j in jobs)))
            def actual(i, s, m, u=u):
                pool = pools[m]
                return planned[i][s] * float(np.exp(np.quantile(pool, u[i, s])))
            for pol in res:
                _, done = simulate(jobs, self.by_type, cal, pol, planned, pred, actual, transfer_min=self.transfer, seed=rep)
                res[pol].append(summarise(jobs, done, cal))
            if progress: progress((rep + 1) / reps)
        keys = [k for k in res["prodiq"][0] if k != "orders"]
        table = []
        for k in keys:
            a = np.array([x[k] for x in res["prodiq"]]); b = np.array([x[k] for x in res["conventional"]])
            d = a - b; m = float(d.mean())
            h_ = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 and d.std() > 0 else 0.0
            table.append(dict(metric=k, conventional=float(b.mean()), prodiq=float(a.mean()), difference=m,
                              ci_low=m - h_, ci_high=m + h_))
        return dict(table=pd.DataFrame(table), has_due=has_due, test_orders=n - i_test,
                    test_from=t_test, runs=reps, test_ops=int(test_rows.sum()))

import types as _types
E = _types.SimpleNamespace(check_machines=check_machines, check_history=check_history, check_orders=check_orders,
                           Factory=Factory, Calendar=Calendar)
# ============================================================================ end of engine

st.set_page_config(page_title="ProdiQ Planner", page_icon="🏭", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = {'machines': 'machine_id,machine_type,available_from\nLATHE-1,TURNING,\nLATHE-2,TURNING,2026-11-02 10:30\nMILL-1,MILLING,\n', 'history': 'order_id,sequence,machine_type,machine_id,part,quantity,run_minutes,start_time,end_time,planned_minutes,due_date,priority\nWO1001,1,TURNING,LATHE-1,Shaft-A,50,184,2026-06-01 06:10,2026-06-01 09:14,170,2026-06-05 18:00,Normal\nWO1001,2,MILLING,MILL-1,Shaft-A,50,95,2026-06-01 10:02,2026-06-01 11:37,90,2026-06-05 18:00,Normal\n', 'open_orders': 'order_id,part,quantity,due_date,route,priority,release_time\nWO2001,Shaft-A,40,2026-11-12 18:00,TURNING > MILLING,Normal,\nWO2002,Flange-B,120,2026-11-10 14:00,MILLING > TURNING > MILLING,Urgent,2026-11-03 08:00\n'}
STUDY_REFERENCE = 'speed,noise,spread_rms,conventional,prodiq,gain,gain_lo,gain_hi\n0.0,0.5,0.0,72.4516,72.9677,0.5161,-0.3789,1.4112\n0.5,0.5,0.1328,73.5593,75.9153,2.3559,0.998,3.7139\n1.0,0.5,0.2655,72.7768,77.8705,5.0937,4.6047,5.5827\n1.5,0.5,0.3982,70.2746,78.0614,7.7868,6.5713,9.0022\n2.0,0.5,0.531,68.7154,79.8211,11.1057,9.9118,12.2996\n0.0,1.0,0.0,72.7307,72.9567,0.226,-0.8268,1.2788\n0.5,1.0,0.1328,73.7848,74.8609,1.0761,0.1256,2.0266\n1.0,1.0,0.2655,76.5537,78.7759,2.2222,1.2367,3.2077\n1.5,1.0,0.3982,76.422,81.4312,5.0092,3.9663,6.052\n2.0,1.0,0.531,78.4586,84.2293,5.7707,4.5612,6.9801\n0.0,1.5,0.0,87.5957,87.5532,-0.0426,-0.6875,0.6024\n0.5,1.5,0.1328,78.5052,79.6907,1.1856,0.2732,2.0979\n1.0,1.5,0.2655,76.3678,78.5747,2.2069,0.6503,3.7635\n1.5,1.5,0.3982,81.9713,83.6756,1.7043,0.523,2.8856\n2.0,1.5,0.531,78.6885,82.9098,4.2213,3.0023,5.4403\n'
BLUE, ORANGE, RED, INK, MUTED = "#2a78d6", "#eb6834", "#e34948", "#52514e", "#8a8984"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SAMPLES = {
    "Sample: machining job shop": dict(dir="sample_jobshop", start=datetime(2026, 3, 1, 6, 0), transfer=30,
                                       cal={6: (6, 22), 0: (6, 22), 1: (6, 22), 2: (6, 22), 3: (6, 22), 4: (6, 14)}),
    "Sample: fabrication shop": dict(dir="sample_fabrication", start=datetime(2026, 9, 1, 7, 0), transfer=20,
                                     cal={d: (7, 19) for d in range(6)}),
}

# ----------------------------------------------------------------------------- sidebar: data + settings
st.sidebar.title("🏭 ProdiQ Planner")
src = st.sidebar.radio("Factory data", list(SAMPLES) + ["Upload my factory"], index=0,
                       help="Try a sample first, then upload your own three files.")
if src == "Upload my factory":
    up_m = st.sidebar.file_uploader("1. Machines (CSV or Excel)", type=["csv", "xlsx", "xls"])
    up_h = st.sidebar.file_uploader("2. Finished operations / history", type=["csv", "xlsx", "xls"])
    up_o = st.sidebar.file_uploader("3. Open orders to plan", type=["csv", "xlsx", "xls"])
    cfg = dict(start=(datetime.now() + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0), transfer=0,
               cal={d: (6, 22) for d in range(5)})
    if up_o is not None:            # old test files: start the plan when the orders start, not today
        try:
            _o = E.check_orders(io.BytesIO(up_o.getvalue())).table
            if _o is not None and len(_o) and _o.due_date.max() < pd.Timestamp(cfg["start"]):
                rel = _o.release_time.min() if "release_time" in _o and _o.release_time.notna().any() else _o.due_date.min() - pd.Timedelta(days=14)
                cfg["start"] = pd.Timestamp(rel).floor("h").to_pydatetime()
                st.sidebar.info(f"All due dates are in the past, so the plan starts {cfg['start']:%d %b %Y %H:%M} (change under Plan start).")
        except Exception:
            pass
else:
    cfg = SAMPLES[src]
    p = os.path.join(HERE, cfg["dir"])
    up_m, up_h, up_o = (f"{p}_machines.csv", f"{p}_history.csv", f"{p}_open_orders.csv")
    _missing = [os.path.basename(f) for f in (up_m, up_h, up_o) if not os.path.exists(f)]
    if _missing:
        st.error("The sample files are missing from the app's folder: " + ", ".join(_missing)
                 + ". Upload them to the GitHub repository next to app.py, or choose **Upload my factory**.")
        st.stop()

with st.sidebar.expander("Working hours", expanded=False):
    cal_df = pd.DataFrame([dict(day=DAYS[d], working=d in cfg["cal"], start=float(cfg["cal"].get(d, (6, 22))[0]),
                                end=float(cfg["cal"].get(d, (6, 22))[1])) for d in range(7)])
    cal_df = st.data_editor(cal_df, hide_index=True, disabled=["day"], key=f"cal_{src}",
                            column_config={"start": st.column_config.NumberColumn("from (h)", min_value=0, max_value=24, step=0.5),
                                           "end": st.column_config.NumberColumn("to (h)", min_value=0, max_value=24, step=0.5)})
    weekly = {DAYS.index(r.day): [(float(r.start), float(r.end))] for r in cal_df.itertuples() if r.working and r.end > r.start}
    transfer = st.number_input("Minutes to move a job between machines", 0, 600, int(cfg["transfer"]), step=5, key=f"tr_{src}")
with st.sidebar.expander("Plan start", expanded=False):
    sk = f"{src}_{cfg['start']:%Y%m%d%H%M}"
    d0 = st.date_input("Date", cfg["start"].date(), key=f"d_{sk}")
    t0 = st.time_input("Time", cfg["start"].time(), key=f"t_{sk}")
plan_start = pd.Timestamp(datetime.combine(d0, t0))
st.sidebar.caption("Machine-aware planning from the ProdiQ research project (VIT). "
                   "Predictions decide **which machine** runs each job; the queue order stays a proven rule.")


# ----------------------------------------------------------------------------- loading (cached)
def _bytes(f):
    if f is None: return None
    if isinstance(f, str): return open(f, "rb").read()
    return f.getvalue()

def _named(b, name):
    x = io.BytesIO(b); x.name = name; return x

@st.cache_resource(show_spinner=False, max_entries=6)
def load_factory(bm, bh, nm, nh, weekly_key, transfer):
    m = E.check_machines(_named(bm, nm)); h = E.check_history(_named(bh, nh), m.table if m.ok else None)
    if not (m.ok and h.ok): return m, h, None
    F = E.Factory(m.table, h.table, {d: [w] for d, w in weekly_key}, transfer).learn()
    return m, h, F

@st.cache_data(show_spinner=False, max_entries=6)
def run_backtest(_F, key):
    return _F.backtest()

bm, bh, bo = _bytes(up_m), _bytes(up_h), _bytes(up_o)
st.title("ProdiQ Planner")
if bm is None or bh is None:
    st.info("Upload your **machines** and **history** files in the sidebar (and **open orders** to get a plan), "
            "or pick a sample factory. The **Your data** tab explains the format and has blank templates.")
    tabs = st.tabs(["📁 Your data"])
    data_tab = tabs[0]
    F = None
else:
    weekly_key = tuple(sorted((d, w[0]) for d, w in weekly.items()))
    if not weekly:
        st.error("Set at least one working day under **Working hours**."); st.stop()
    with st.spinner("Learning how long jobs take on each machine… (about 20 seconds, once per data set)"):
        mres, hres, F = load_factory(bm, bh, getattr(up_m, "name", str(up_m)), getattr(up_h, "name", str(up_h)),
                                     weekly_key, float(transfer))
    ores = E.check_orders(_named(bo, getattr(up_o, "name", str(up_o))), mres.table if mres.ok else None) if bo else None
    tabs = st.tabs(["📋 Plan", "⚙️ Machines", "🧪 Will it help here?", "🎯 Prediction accuracy", "📁 Your data"])
    plan_tab, mach_tab, test_tab, acc_tab, data_tab = tabs
    data_key = hashlib.md5((bm or b"") + (bh or b"")).hexdigest() + str(weekly_key) + str(transfer)

# ----------------------------------------------------------------------------- tab: your data
with data_tab:
    st.subheader("What ProdiQ needs from your factory")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Machines** – one row per machine\n\n- `machine_id` – e.g. LATHE-2\n- `machine_type` – the kind of "
                    "work it does; machines of the same type can replace each other\n- *optional* `available_from` – "
                    "when a busy machine becomes free")
        st.download_button("Template", TEMPLATES["machines"].encode(), "machines_template.csv")
    with c2:
        st.markdown("**2. History** – one row per finished operation (a few months or more)\n\n- `order_id`, `machine_type`, "
                    "`machine_id`, `part`, `quantity`\n- `run_minutes`, or `start_time` + `end_time`\n- *optional* "
                    "`start_time` (needed for the *Will it help here?* test), `planned_minutes`, `due_date`, `priority`, `sequence`")
        st.download_button("Template", TEMPLATES["history"].encode(), "history_template.csv")
    with c3:
        st.markdown("**3. Open orders** – what needs to be made\n\n- `order_id`, `part`, `quantity`, `due_date`\n- `route` – "
                    "the machine types still to visit, in order, e.g. `TURNING > MILLING > INSPECT`\n- *optional* "
                    "`priority` (Normal / High / Urgent), `release_time`")
        st.download_button("Template", TEMPLATES["open_orders"].encode(), "open_orders_template.csv")
    st.caption("Column names don't have to match exactly: common names such as *Work Center*, *Machine*, *Job No*, *Qty*, "
               "*Product* or *Routing* are recognised. CSV (comma, semicolon or tab) and Excel files both work.")
    if F is not None or (bm is not None and bh is not None):
        st.subheader("Checks on the files you gave")
        for label, r in [("Machines", mres), ("History", hres), ("Open orders", ores)]:
            if r is None: st.write(f"**{label}:** not provided."); continue
            if r.ok: st.success(f"**{label}:** OK – {len(r.table):,} rows.")
            for e in r.errors: st.error(f"**{label}:** {e}")
            for w in r.warnings: st.warning(f"**{label}:** {w}")
            for n in r.notes: st.caption(n)

if F is None:
    if bm is not None and bh is not None:
        with tabs[0]: st.error("The files have problems that stop ProdiQ from learning. See **Your data** for details.")
    st.stop()

# ----------------------------------------------------------------------------- tab: plan
P = None
with plan_tab:
    if ores is None:
        st.info("Add an **open orders** file to get a plan.")
    elif not ores.ok:
        for e in ores.errors: st.error(e)
    else:
        with st.spinner("Planning…"):
            try:
                P = F.plan(ores.table, plan_start)
            except ValueError as e:
                st.error(f"Could not build the plan: {e}"); st.stop()
        kp, kc = P["kpi_prodiq"], P["kpi_conventional"]
        st.caption(f"Plan from **{plan_start:%a %d %b %Y, %H:%M}** for {kp['orders']} open orders. Both plans use the same queue rule; "
                   "ProdiQ also picks the machine expected to finish each job fastest. *Usual way* = any free machine "
                   "(average of 20 random choices).")
        c = st.columns(4)
        c[0].metric("On time – ProdiQ plan", f"{kp['on_time']:.0%}", f"{(kp['on_time'] - kc['on_time']) * 100:+.1f} pts vs usual way")
        c[1].metric("Late orders – ProdiQ plan", f"{kp['late_orders']}", f"{kp['late_orders'] - kc['late_orders']:+.1f} vs usual way",
                    delta_color="inverse")
        c[2].metric("Average order time", f"{kp['mean_lead_h']:.0f} h", f"{kp['mean_lead_h'] - kc['mean_lead_h']:+.1f} h vs usual way",
                    delta_color="inverse")
        c[3].metric("All orders done by", f"{P['end']:%d %b}", f"{P['end']:%A, %H:%M}", delta_color="off", delta_arrow="off")
        if kp["on_time"] < kc["on_time"] - 1e-9:
            st.info("In this particular plan, choosing machines the usual way comes out slightly ahead. With a few dozen "
                    "orders, a few points either way is normal. The **Will it help here?** test, which replays months of "
                    "your history, is the reliable guide to whether ProdiQ's machine choice helps your factory.")
        od = P["orders"].copy()
        risk = od[od.late_prodiq | (od.hours_early_late > -8)].sort_values("hours_early_late", ascending=False)
        st.subheader("Orders to watch")
        if risk.empty: st.success("Every order is planned to finish at least 8 working hours before its due date.")
        else:
            st.dataframe(risk[["order_id", "part", "quantity", "priority", "due_date", "finish_prodiq", "hours_early_late", "route"]],
                         hide_index=True, column_config={
                             "hours_early_late": st.column_config.NumberColumn("hours late (+) / early (−)", format="%.1f"),
                             "finish_prodiq": st.column_config.DatetimeColumn("planned finish", format="ddd D MMM, HH:mm"),
                             "due_date": st.column_config.DatetimeColumn("due", format="ddd D MMM, HH:mm")})
        st.subheader("Machine schedule")
        g = P["gantt"].merge(od[["order_id", "late_prodiq", "priority"]], on="order_id")
        g["status"] = np.where(g.late_prodiq, "Order will be late", "Order on time")
        types = sorted(g.machine_type.unique())
        pick = st.multiselect("Machine types", types, default=types)
        gg = g[g.machine_type.isin(pick)]
        if len(gg):
            fig = px.timeline(gg, x_start="start", x_end="end", y="machine_id", color="status",
                              color_discrete_map={"Order on time": BLUE, "Order will be late": RED},
                              hover_data={"order_id": True, "step": True, "minutes": ":.0f", "priority": True,
                                          "machine_id": False, "status": False},
                              category_orders={"machine_id": sorted(gg.machine_id.unique())})
            fig.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,0.9)")
            fig.update_layout(height=max(320, 26 * gg.machine_id.nunique() + 120), legend_title_text="",
                              legend=dict(orientation="h", y=1.05, x=0), margin=dict(l=0, r=0, t=30, b=0),
                              yaxis_title=None, xaxis_title=None)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, key="gantt")
        st.download_button("Download the plan (one row per operation)", P["gantt"].to_csv(index=False).encode(),
                           "prodiq_plan.csv", type="primary")
        st.download_button("Download order summary", od.to_csv(index=False).encode(), "prodiq_orders.csv")
        with st.expander("All orders"):
            st.dataframe(od, hide_index=True)
        st.caption("The plan assumes every machine is free at the plan start unless *available_from* says otherwise, "
                   "and uses ProdiQ's expected time for each job on the machine it is given.")

# ----------------------------------------------------------------------------- tab: machines
with mach_tab:
    sp = F.speeds[F.speeds.machine_type.map(lambda t: len(F.by_type[t]) > 1)].copy()
    st.subheader("Which machines are faster?")
    st.caption("For the jobs each machine type really ran: how much longer (+) or shorter (−) ProdiQ expects each machine "
               "to take than the average machine of its type. The dot is what the history shows for jobs that machine actually did.")
    if sp.empty:
        st.info("No machine type has more than one machine, so there is no machine choice for ProdiQ to make.")
    else:
        sp["label"] = sp.machine_type + " · " + sp.machine_id
        sp = sp.sort_values(["machine_type", "machine_id"])
        fig = go.Figure()
        fig.add_bar(y=sp.label, x=sp.predicted_vs_type_avg * 100, orientation="h", name="ProdiQ expects", marker_color=BLUE,
                    hovertemplate="%{y}: %{x:+.0f}%<extra>ProdiQ expects</extra>")
        fig.add_scatter(y=sp.label, x=sp.observed_vs_type_avg * 100, mode="markers", name="History shows",
                        marker=dict(color=INK, size=9, symbol="diamond"), hovertemplate="%{y}: %{x:+.0f}%<extra>History shows</extra>")
        fig.update_layout(height=max(300, 24 * len(sp) + 100), xaxis_title="% time vs type average (− = faster)",
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.06, x=0), margin=dict(l=0, r=0, t=30, b=0))
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
        st.plotly_chart(fig, key="speeds")
        big = F.spread.sort_values(ascending=False)
        top = big.index[0]
        st.markdown(f"**Biggest speed gap:** in **{top}**, the slowest machine is expected to take "
                    f"**{big.iloc[0]:.0%}** longer (as a share of the type average) than the fastest. The bigger these gaps, "
                    "the more ProdiQ's machine choice matters.")
        ref = pd.read_csv(io.StringIO(STUDY_REFERENCE))
        lv = ref.groupby("speed").agg(spread=("spread_rms", "first"), lo=("gain", "min"), hi=("gain", "max")).reset_index()
        near = lv.iloc[(lv.spread - F.spread_rms).abs().argmin()]
        st.subheader("How much could ProdiQ help a factory like yours?")
        st.markdown(f"A typical machine in your factory runs about **{np.expm1(F.spread_rms):.0%}** faster or slower than the "
                    f"average machine of its type. The ProdiQ study tested 15 simulated factories with different speed gaps "
                    f"and different amounts of job-to-job randomness. Factories with speed gaps closest to yours gained "
                    f"**{near.lo:+.1f} to {near.hi:+.1f} points** of on-time delivery.")
        tb = pd.DataFrame({"Typical machine speed gap": [f"{np.expm1(v):.0%}" for v in lv.spread],
                           "On-time gain in the study (points)": [f"{round(a_, 1) + 0.0:+.1f} to {round(b_, 1) + 0.0:+.1f}" for a_, b_ in zip(lv.lo, lv.hi)],
                           "": ["← closest to your factory" if v == near.speed else "" for v in lv.speed]})
        st.dataframe(tb, hide_index=True)
        st.caption("Gains vs planning the usual way, 10 paired replays per factory, at the study's normal workload. "
                   "Use the *Will it help here?* test for your own factory's answer.")
    if P is not None:
        st.subheader("Hours of work per machine in the plan")
        mh = P["machine_hours"].rename("hours").reset_index()
        mh = mh.merge(F.machines[["machine_id", "machine_type"]], on="machine_id").sort_values(["machine_type", "machine_id"])
        fig2 = px.bar(mh, x="machine_id", y="hours", color_discrete_sequence=[BLUE], hover_data=["machine_type"])
        fig2.update_layout(height=320, xaxis_title=None, yaxis_title="planned hours", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, key="mhours")

# ----------------------------------------------------------------------------- tab: will it help here?
NICE = {"on_time": ("Orders on time", "pct", 1), "late_orders": ("Late orders", "n", -1),
        "mean_tardiness_h": ("Average delay per order (hours)", "h", -1), "urgent_on_time": ("Urgent orders on time", "pct", 1),
        "mean_lead_h": ("Average order time, release to finish (hours)", "h", -1),
        "p95_lead_h": ("Slowest 5% of orders take at least (hours)", "h", -1)}
with test_tab:
    st.subheader("Would ProdiQ have helped your factory?")
    st.markdown("ProdiQ replays the **most recent quarter of your history** twice – once choosing machines the usual way, "
                "once with ProdiQ – with the model trained only on the months *before* that period. How long each job takes "
                "on each machine is drawn from what that machine really did, so the test does not rely on ProdiQ's own "
                "predictions being right. It is repeated 10 times with different random draws.")
    if st.button("Run the test (about 20 seconds)", type="primary") or st.session_state.get("bt_done") == data_key:
        with st.spinner("Replaying your history…"):
            bt = run_backtest(F, data_key)
        st.session_state["bt_done"] = data_key
        if "error" in bt:
            st.warning(bt["error"])
        else:
            t = bt["table"].set_index("metric")
            main = "on_time" if bt["has_due"] else "mean_lead_h"
            r = t.loc[main]; sign = NICE[main][2]
            good = (r.ci_low > 0) if sign > 0 else (r.ci_high < 0)
            bad = (r.ci_high < 0) if sign > 0 else (r.ci_low > 0)
            if good: st.success("**Yes – ProdiQ helped in this test.** Use the ProdiQ plan.")
            elif bad: st.error("**No – ProdiQ made things worse in this test.** Plan the usual way for this factory and check "
                               "the machine data (are machine IDs recorded correctly?).")
            else: st.info("**No clear difference.** ProdiQ is unlikely to hurt, but the gain is too small to see here. "
                          "This is common when machines of the same type run at similar speeds.")
            rows = []
            for k, (name, kind, _) in NICE.items():
                if k not in t.index: continue
                x = t.loc[k]
                f = (lambda v: f"{v:.1%}") if kind == "pct" else (lambda v: f"{v:,.1f}")
                fd = (lambda v: f"{v * 100:+.1f} pts") if kind == "pct" else (lambda v: f"{v:+,.1f}")
                rows.append({"": name, "Usual way": f(x.conventional), "ProdiQ": f(x.prodiq), "Difference": fd(x.difference),
                             "95% range of difference": f"{fd(x.ci_low)} to {fd(x.ci_high)}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(f"Test period: orders released from {bt['test_from']:%d %b %Y} ({bt['test_orders']} orders, "
                       f"{bt['test_ops']:,} operations), averaged over {bt['runs']} replays. "
                       + ("" if bt["has_due"] else "Your history has no due dates, so the test compares order times instead of on-time delivery. "))

# ----------------------------------------------------------------------------- tab: accuracy
with acc_tab:
    st.subheader("How good are ProdiQ's time predictions?")
    if F.accuracy is None:
        st.info("Not enough history to hold some back for checking.")
    else:
        a = F.accuracy
        b0, b1 = a["Standard times (conventional)"], a["ProdiQ prediction"]
        c = st.columns(3)
        c[0].metric("Typical error – standard times", f"{b0['mae']:.0f} min", help="Average size of the miss, in minutes, on the most recent 20% of history")
        c[1].metric("Typical error – ProdiQ", f"{b1['mae']:.0f} min", f"{b1['mae'] - b0['mae']:+.0f} min", delta_color="inverse")
        if F.agreement:
            c[2].metric("Knows which machine is faster", f"{F.agreement[0]:.0%}",
                        help=f"Of {F.agreement[1]} pairs of same-type machines, the share where ProdiQ's ranking matched what really happened")
        st.markdown("Single jobs are hard to predict exactly – the same job can take half or double the usual time. What matters "
                    "for ProdiQ is the last number: whether it knows **which machine is faster**. That is why ProdiQ uses its "
                    "predictions to choose machines, not to decide when to start or hold jobs.")
        st.caption("Checked on the most recent 20% of your history, with the model trained only on the earlier 80%.")
