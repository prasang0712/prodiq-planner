# ProdiQ Planner (web app)

ProdiQ plans a job shop's open orders. It learns from the factory's own history how long jobs take **on each machine**,
then sends each job to the free machine that should finish it fastest. Queue order stays with a proven rule
(Apparent Tardiness Cost). This is the configuration that won the ProdiQ research benchmark: in the full-year
simulation it raised on-time delivery from 75.9% to 78.7%, and urgent orders from 62.6% to 68.7%.

It works with **any factory's data**. You give it three files and it does the rest.

| File | One row per | Must have | Nice to have |
|---|---|---|---|
| Machines | machine | `machine_id`, `machine_type` | `available_from` |
| History | finished operation | `order_id`, `machine_type`, `machine_id`, `part`, `quantity`, and `run_minutes` *or* `start_time` + `end_time` | `start_time`, `planned_minutes`, `due_date`, `priority`, `sequence` |
| Open orders | order to make | `order_id`, `part`, `quantity`, `due_date`, `route` (e.g. `CUT > WELD > PAINT`) | `priority`, `release_time` |

Blank templates are in `templates/`. Common column names (*Work Center*, *Machine*, *Job No*, *Qty*, *Product*,
*Routing*…) are recognised automatically, and CSV and Excel files both work.

## What the app shows
- **Plan**: the machine schedule (Gantt chart), orders at risk of being late, and a comparison with planning the usual
  way. The plan can be downloaded as a CSV.
- **Machines**: which machines are faster or slower than others of their type.
- **Will it help here?**: replays the last quarter of your history with and without ProdiQ and tells you, with a
  95% range, whether ProdiQ would have helped. Run this before trusting it in a new factory.
- **Prediction accuracy**: how well ProdiQ predicts job times, and how often it knows which machine is faster.

## Put it online (free, about 10 minutes, no coding)
1. Make a free account on **github.com**. Create a new repository and upload everything in this folder (drag the files
   into the browser). The folder is small enough for the web uploader.
2. Go to **share.streamlit.io**, sign in with GitHub, click **Create app**, pick your repository, and set the main file
   to `app.py`. Click **Deploy**.
3. After a few minutes you get a link like `https://your-name-prodiq.streamlit.app` that anyone can open.

## Run it on your own computer
1. Install Python (3.10 or newer) from python.org.
2. Open a terminal in this folder and run `pip install -r requirements.txt`.
3. Run `streamlit run app.py`. The app opens in your browser.

## What's inside
- `app.py`: the web page.
- `prodiq_engine/`: the planning engine (reading and checking files, the prediction model, the scheduler and the
  back-test). Other software can use it directly. Your CS friend can call `Factory(...).learn().plan(...)` from a
  FastAPI backend.
- `sample_data/`: two example factories. *job_shop* is the research factory, built to match a real production log
  (Levy 2014). *fabrication_shop* is a made-up sheet-metal shop with different machines, calendar and column names.
- `tests/`: automatic checks (`python -m pytest -q tests`).
- `tools/`: scripts that built the sample data.

## Limits to know
- The plan assumes machines are free at the plan start unless `available_from` says otherwise. It does not model
  breakdowns, set-ups between part families, or operator availability.
- Predictions use machine type, machine, part, quantity and standard time. More information (operator, tool wear,
  sensor data) is not used yet.
- ProdiQ only helps where machines of the same type really differ in speed. The *Will it help here?* test checks
  that for your factory.
