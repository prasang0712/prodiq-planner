"""Run with:  python -m pytest -q tests"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prodiq_engine as E
from prodiq_engine.plan import Calendar, simulate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def brute_add(cal, t, work):
    """minute-by-minute reference for Calendar.add"""
    t = cal.next_working(t)
    step = 0.25
    while work > 1e-9:
        if cal.next_working(t) != t: t = cal.next_working(t); continue
        d = min(step, work); t += d; work -= d
    return t


def test_calendar_matches_brute_force():
    cal = Calendar("2026-03-06 13:00", {6: [(6, 22)], 0: [(6, 22)], 4: [(6, 14)]}, horizon_days=60)  # Fri afternoon
    rng = np.random.default_rng(1)
    for _ in range(25):
        t, w = float(rng.uniform(0, 5000)), float(rng.uniform(1, 2500))
        assert abs(cal.add(t, w) - brute_add(cal, t, w)) < 0.5


def test_calendar_skips_off_days():
    cal = Calendar("2026-03-06 13:00", {4: [(6, 14)], 6: [(6, 22)]})   # Fri 06-14, Sun 06-22
    end = cal.add(0, 120)                                              # 1h left on Friday, 1h on Sunday
    assert cal.ts(end) == pd.Timestamp("2026-03-08 07:00")


def test_column_synonyms():
    df = pd.DataFrame({"Machine": ["A1", "A2"], "Work Center": ["CUT", "CUT"]})
    r = E.check_machines(df)
    assert r.ok and list(r.table.columns[:2]) == ["machine_id", "machine_type"]


def test_orders_reject_unknown_machine_type():
    m = E.check_machines(pd.DataFrame({"machine_id": ["A1"], "machine_type": ["CUT"]})).table
    o = E.check_orders(pd.DataFrame({"order_id": ["O1"], "part": ["P"], "quantity": [5], "due_date": ["2026-11-01"],
                                     "route": ["CUT > PAINT"]}), m)
    assert not o.ok and "PAINT" in o.errors[0]


def test_prodiq_picks_faster_machine():
    by_type = {"CUT": ["SLOW", "FAST"]}
    cal = Calendar("2026-03-02 00:00", {d: [(0, 24)] for d in range(7)})
    jobs = [dict(id="J1", route=["CUT"], release=0.0, due=1000.0, weight=1.0)]
    pred = [[{"SLOW": 100.0, "FAST": 50.0}]]
    ops, done = simulate(jobs, by_type, cal, "prodiq", [[75.0]], pred, lambda i, s, m: pred[i][s][m])
    assert ops[0][3] == "FAST" and done[0] == 50.0


def test_end_to_end_fabrication_sample():
    d = f"{HERE}/sample_data/fabrication_shop"
    m = E.check_machines(f"{d}/machines.csv"); h = E.check_history(f"{d}/history.csv", m.table)
    o = E.check_orders(f"{d}/open_orders.csv", m.table)
    assert m.ok and h.ok and o.ok
    F = E.Factory(m.table, h.table, {d_: [(7, 19)] for d_ in range(6)}, 20).learn()
    # the sample was generated with CUT-3 20% faster and CUT-1 25% slower than CUT-2; the model should see that
    s = F.speeds.set_index("machine_id").predicted_vs_type_avg
    assert s["CUT-3"] < s["CUT-2"] < s["CUT-1"]
    P = F.plan(o.table, "2026-09-01 07:00")
    assert len(P["orders"]) == len(o.table) and P["gantt"].start.min() >= pd.Timestamp("2026-09-01 07:00")
    # no machine runs two operations at once
    for _, g in P["gantt"].sort_values("start").groupby("machine_id"):
        assert (g.start.iloc[1:].values >= g.end.iloc[:-1].values).all()
    # operations of an order run in route order
    for _, g in P["gantt"].sort_values("step").groupby("order_id"):
        assert (g.start.iloc[1:].values >= g.end.iloc[:-1].values).all()
