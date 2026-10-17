"""Processing-time models.

StandardTimes : the conventional estimate - one time per (machine type, part, quantity), the same on every machine.
                Uses the factory's own planned minutes when they are supplied; otherwise it is estimated from history
                (log time = type level + part effect + quantity slope, part effects shrunk toward zero when data are thin).
TimeModel     : ProdiQ's machine-aware prediction - a Random Forest on machine type, machine, part, quantity and the
                standard time, trained on log minutes (predicts the median) and rescaled per type to the mean
                ("smearing"), exactly as in the research benchmark.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
