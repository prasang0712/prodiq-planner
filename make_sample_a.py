"""Sample factory A: the ProdiQ research job shop (calibrated to the Levy 2014 log), cut at a snapshot date."""
import pandas as pd, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/rep/work/prodiq_dataset"
OUT = "sample_data/job_shop"
SNAP = pd.Timestamp("2026-03-01 06:00")          # a Sunday morning (working day in this shop)
import os; os.makedirs(OUT, exist_ok=True)
op = pd.read_csv(f"{SRC}/operations.csv", parse_dates=["start_ts", "end_ts"])
wo = pd.read_csv(f"{SRC}/work_orders.csv", parse_dates=["release_ts", "due_ts"])
mc = pd.read_csv(f"{SRC}/machines.csv")
op = op[op.is_warmup == 0].merge(wo[["wo_id", "due_ts"]], on="wo_id")
mach = mc[["machine_id", "op_type", "op_type_name"]].rename(columns={"op_type": "machine_type", "op_type_name": "machine_type_name"})
running = op[(op.start_ts < SNAP) & (op.end_ts > SNAP)].set_index("machine_id").end_ts
mach["available_from"] = mach.machine_id.map(running).dt.round("min")
mach.to_csv(f"{OUT}/machines.csv", index=False)
hist = op[op.end_ts <= SNAP].rename(columns={"wo_id": "order_id", "seq": "sequence", "op_type": "machine_type",
       "qty_in": "quantity", "std_time_min": "planned_minutes", "actual_time_min": "run_minutes",
       "start_ts": "start_time", "end_ts": "end_time", "due_ts": "due_date"})
hist = hist[["order_id", "sequence", "machine_type", "machine_id", "part", "quantity", "priority", "planned_minutes",
             "run_minutes", "start_time", "end_time", "due_date"]].sort_values("start_time")
hist.to_csv(f"{OUT}/history.csv", index=False)
# open orders: released before the snapshot (or in the next 14 days) and not finished; route = operations not yet started
cand = wo[(wo.release_ts < SNAP + pd.Timedelta(days=14)) & (pd.to_datetime(wo.completion_ts) > SNAP)]
rows = []
for r in cand.itertuples():
    o = op[(op.wo_id == r.wo_id) & (op.start_ts >= SNAP) & (~op.is_rework)].sort_values("seq")
    if o.empty: continue
    rows.append(dict(order_id=r.wo_id, part=r.part, quantity=int(o.qty_in.iloc[0]), priority=r.priority,
                     release_time=max(r.release_ts, SNAP).round("min"), due_date=r.due_ts.round("min"),
                     route=" > ".join(o.op_type)))
pd.DataFrame(rows).to_csv(f"{OUT}/open_orders.csv", index=False)
print(len(mach), "machines;", len(hist), "history rows;", len(rows), "open orders")
