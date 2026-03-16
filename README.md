# Two-Queue Merge System Control Applet

Interactive Streamlit app for exploring a two-queue / one-server merge system under several control policies:

- `none`
- `priority`
- `threshold_priority`
- `rate_throttling`
- `hybrid`

## Features

- Editable inputs for `λ₁`, `λ₂`, and `μ`
- Policy dropdown
- Policy-specific parameter inputs that appear only when needed
- Queue/server system diagram
- Stability indicator based on nominal load `ρ = (λ₁ + λ₂) / μ`
- Sample-path trajectory plot
- Optional animation
- Compare-all-policies table and chart
- CSV export of the selected trajectory

## Files

- `app.py` — Streamlit interface
- `simulator.py` — DES engine
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py