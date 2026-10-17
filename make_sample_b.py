"""Sample factory B: a made-up sheet-metal fabrication shop with a different layout, calendar and column names,
used to show that ProdiQ is not tied to the research job shop. History is produced by simulating the shop
with conventional dispatching (any free machine)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prodiq_engine.plan import Calendar, simulate, PRIORITY_WEIGHT

rng = np.random.default_rng(7)
OUT = "sample_data/fabrication_shop"; os.makedirs(OUT, exist_ok=True)
EFF = {"CUT-1": .25, "CUT-2": 0.0, "CUT-3": -.20, "BEND-1": .15, "BEND-2": -.10, "WELD-1": .30, "WELD-2": .10,
       "WELD-3": -.10, "WELD-4": -.25, "PAINT-1": 0.0, "INSP-1": 0.0, "INSP-2": 0.0, "PACK-1": 0.0}
TYPE = {m: m.split("-")[0].replace("INSP", "INSPECT") for m in EFF}
BASE = {"CUT": 4.8, "BEND": 4.0, "WELD": 10.5, "PAINT": 3.6, "INSPECT": 1.4, "PACK": 0.8}   # minutes per unit^0.8
NOISE = {"CUT": .25, "BEND": .3, "WELD": .45, "PAINT": .2, "INSPECT": .3, "PACK": .2}
ROUTES = ["CUT>BEND>WELD>PAINT>INSPECT>PACK", "CUT>WELD>INSPECT>PACK", "CUT>BEND>INSPECT>PACK",
          "CUT>WELD>WELD>PAINT>INSPECT>PACK", "CUT>BEND>WELD>INSPECT>PACK", "CUT>PAINT>INSPECT>PACK"]
PARTS = {f"P-{100 + i}": (ROUTES[i % len(ROUTES)], float(rng.normal(0, .25))) for i in range(14)}
WEEK = {d: [(7.0, 19.0)] for d in range(6)}                 # Mon-Sat 07:00-19:00
by_type = {}
for m, t in TYPE.items(): by_type.setdefault(t, []).append(m)

def make_orders(n_days, rate, t_start):
    t, out = 0.0, []
    while True:
        t += rng.exponential(1440 / rate)
        if t > n_days * 1440: return out
        part = rng.choice(list(PARTS)); route, pe = PARTS[part]
        out.append(dict(part=part, route=route.split(">"), qty=int(np.clip(rng.lognormal(np.log(60), .6), 5, 500)),
                        release=t_start + t, pe=pe, priority=rng.choice(["Normal", "High", "Urgent"], p=[.75, .2, .05])))

def std_min(o, ty): return BASE[ty] * o["qty"] ** 0.8 * np.exp(o["pe"])

T0 = pd.Timestamp("2025-09-01 07:00"); cal = Calendar(T0, WEEK, horizon_days=500)
orders = make_orders(365, 6.5, 0.0)
eps = rng.normal(size=(len(orders), 8))
jobs = [dict(id=f"J{i + 1:05d}", route=o["route"], release=o["release"], due=None,
             weight=PRIORITY_WEIGHT[o["priority"]], priority=o["priority"]) for i, o in enumerate(orders)]
planned = [[std_min(o, ty) for ty in o["route"]] for o in orders]
pred = [[{m: p for m in by_type[ty]} for ty, p in zip(o["route"], pl)] for o, pl in zip(orders, planned)]
def actual(i, s, m): return planned[i][s] * float(np.exp(EFF[m] + NOISE[TYPE[m]] * eps[i, s]))
ops, done = simulate(jobs, by_type, cal, "conventional", planned, pred, actual, transfer_min=20)
work = np.array([sum(p) for p in planned]); lead = np.array([d - j["release"] for d, j in zip(done, jobs)])
k = float(np.quantile(lead / work, 0.75))
mult = {"Normal": 1.0, "High": .85, "Urgent": .7}
due = [j["release"] + k * mult[j["priority"]] * w for j, w in zip(jobs, work)]
busy = pd.DataFrame(ops, columns=["i", "s", "t", "m", "a", "b", "d"]).groupby("t").d.sum()
print("utilisation by type:", (busy / (pd.Series({t: len(v) for t, v in by_type.items()}) * cal.week_min * 52)).round(2).to_dict())
print("on time under conventional:", np.mean(np.array(done) <= np.array(due)).round(3))
hist = pd.DataFrame([{"Job No": jobs[i]["id"], "Op": s + 1, "Work Center": t, "Machine": m, "Product": orders[i]["part"],
                      "Qty": orders[i]["qty"], "Priority": orders[i]["priority"],
                      "Start": cal.ts(a).round("min"), "End": cal.ts(b).round("min"), "Run Minutes": round(d, 1),
                      "Due Date": cal.ts(due[i]).round("min")} for i, s, t, m, a, b, d in ops]).sort_values("Start")
snap = cal.ts(365 * 1440).normalize() + pd.Timedelta(hours=7)
hist = hist[hist.End <= snap]
hist.to_csv(f"{OUT}/history.csv", index=False)
pd.DataFrame([{"Machine": m, "Work Center": t} for m, t in TYPE.items()]).to_csv(f"{OUT}/machines.csv", index=False)
new = make_orders(10, 6.5, 0.0)
cal2 = Calendar(snap, WEEK)
op_rows = []
for i, o in enumerate(new):
    w = sum(std_min(o, ty) for ty in o["route"])
    rel = cal2.ts(o["release"]).round("min")
    op_rows.append({"Job No": f"N{i + 1:04d}", "Product": o["part"], "Qty": o["qty"], "Priority": o["priority"],
                    "Release": rel, "Due Date": (cal2.ts(o["release"] + k * mult[o["priority"]] * w)).round("min"),
                    "Routing": " > ".join(o["route"])})
pd.DataFrame(op_rows).to_csv(f"{OUT}/open_orders.csv", index=False)
print(len(hist), "history rows,", len(op_rows), "open orders, snapshot", snap)
