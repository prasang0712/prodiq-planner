"""Reading and checking a factory's files.

Three tables describe a factory:
  machines  - one row per machine: machine_id, machine_type (optional: available_from)
  history   - one row per finished operation: order_id, machine_type, machine_id, part, quantity,
              and run_minutes or start_time + end_time (optional: planned_minutes, due_date, priority, sequence)
  orders    - one row per open order: order_id, part, quantity, due_date, route (machine types joined by ">")
              (optional: priority, release_time)
Column names are matched loosely (case, spaces and common synonyms), so most factory exports load as they are.
"""
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

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
PRIORITY_WEIGHT = {"normal": 1.0, "high": 2.0, "urgent": 4.0}
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
