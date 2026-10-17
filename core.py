"""ProdiQ for any factory: learn from history, plan open orders, and test whether ProdiQ helps this factory."""
import numpy as np
import pandas as pd
from scipy import stats

from .model import StandardTimes, TimeModel, accuracy, machine_speeds, ranking_agreement
from .plan import Calendar, simulate, summarise, PRIORITY_WEIGHT


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
