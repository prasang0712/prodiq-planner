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

import prodiq_engine as E

st.set_page_config(page_title="ProdiQ Planner", page_icon="🏭", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, RED, INK, MUTED = "#2a78d6", "#eb6834", "#e34948", "#52514e", "#8a8984"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SAMPLES = {
    "Sample: machining job shop": dict(dir="sample_data/job_shop", start=datetime(2026, 3, 1, 6, 0), transfer=30,
                                       cal={6: (6, 22), 0: (6, 22), 1: (6, 22), 2: (6, 22), 3: (6, 22), 4: (6, 14)}),
    "Sample: fabrication shop": dict(dir="sample_data/fabrication_shop", start=datetime(2026, 9, 1, 7, 0), transfer=20,
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
    up_m, up_h, up_o = (f"{p}/machines.csv", f"{p}/history.csv", f"{p}/open_orders.csv")

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
        st.download_button("Template", open(f"{HERE}/templates/machines_template.csv", "rb").read(), "machines_template.csv")
    with c2:
        st.markdown("**2. History** – one row per finished operation (a few months or more)\n\n- `order_id`, `machine_type`, "
                    "`machine_id`, `part`, `quantity`\n- `run_minutes`, or `start_time` + `end_time`\n- *optional* "
                    "`start_time` (needed for the *Will it help here?* test), `planned_minutes`, `due_date`, `priority`, `sequence`")
        st.download_button("Template", open(f"{HERE}/templates/history_template.csv", "rb").read(), "history_template.csv")
    with c3:
        st.markdown("**3. Open orders** – what needs to be made\n\n- `order_id`, `part`, `quantity`, `due_date`\n- `route` – "
                    "the machine types still to visit, in order, e.g. `TURNING > MILLING > INSPECT`\n- *optional* "
                    "`priority` (Normal / High / Urgent), `release_time`")
        st.download_button("Template", open(f"{HERE}/templates/open_orders_template.csv", "rb").read(), "open_orders_template.csv")
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
        ref = pd.read_csv(os.path.join(HERE, "prodiq_engine", "study_reference.csv"))
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
