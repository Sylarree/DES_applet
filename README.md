# Two-Queue Assembly System Control Applet

Interactive Streamlit app for exploring a **two-queue assembly/synchronization system**.

In this model:
- Queue 1 and Queue 2 receive arrivals separately
- one item from Queue 1 and one item from Queue 2 are joined
- the assembled job is then processed by a server

## Implemented control policies

- `none`
- `hard_blocking`
- `rate_throttling`
- `probabilistic_acceptance`
- `hybrid`

## Features

- editable λ₁, λ₂, and μ
- policy-specific inputs that appear only when needed
- system schematic showing assembly before service
- live or static sample-path plots
- diagnostics for queue lengths, imbalance, throughput, utilization
- compare-all-policies table and chart
- CSV export of simulated trajectories

## Files

- `app.py` — Streamlit interface
- `simulator.py` — DES engine
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py