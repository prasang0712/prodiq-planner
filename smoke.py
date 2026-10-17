import time, sys
sys.path.insert(0, "/home/claude/prodiq_app")
import prodiq_engine as E
D = sys.argv[1] if len(sys.argv) > 1 else "sample_data/job_shop"
cal = {6: [(6, 22)], 0: [(6, 22)], 1: [(6, 22)], 2: [(6, 22)], 3: [(6, 22)], 4: [(6, 14)]}
if len(sys.argv) > 2: cal = {d: [(7, 19)] for d in range(6)}
m = E.check_machines(f"{D}/machines.csv"); print("machines", m.ok, m.errors, m.warnings, m.notes)
h = E.check_history(f"{D}/history.csv", m.table); print("history", h.ok, h.errors, h.warnings, h.notes)
o = E.check_orders(f"{D}/open_orders.csv", m.table); print("orders", o.ok, o.errors, o.warnings, o.notes)
t = time.time(); F = E.Factory(m.table, h.table, cal).learn(); print("learn", round(time.time() - t, 1))
print(F.accuracy, F.agreement); print(F.spread.round(3).to_dict())
print(F.speeds.head(8).round(3).to_string())
t = time.time(); p = F.plan(o.table, o.table.release_time.min() if "release_time" in o.table else "2026-03-01 06:00"); print("plan", round(time.time() - t, 1))
print(p["kpi_prodiq"], p["kpi_conventional"]); print(p["gantt"].head(3).to_string())
t = time.time(); b = F.backtest(); print("backtest", round(time.time() - t, 1))
print(b.get("error")) if "error" in b else print(b["table"].round(3).to_string(), b["test_orders"], b["test_from"])
