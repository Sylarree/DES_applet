import time as pytime
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

from simulator import simulate_assembly_system, compare_policies

st.set_page_config(page_title="Assembly System Control Applet", layout="wide")

st.title("Two-Queue Assembly System Control Applet")
st.caption(
    "Interactive DES simulator for an assembly/synchronization system: one item from Queue 1 and one item from Queue 2 are joined before service."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Model Parameters")

policy = st.sidebar.selectbox(
    "Control Policy",
    ["none", "hard_blocking", "rate_throttling", "probabilistic_acceptance", "hybrid"],
    index=0,
)

policy_help = {
    "none": "Baseline assembly system with no control.",
    "hard_blocking": "Block arrivals to the longer queue once imbalance exceeds threshold.",
    "rate_throttling": "Reduce arrival rate of the longer queue when imbalance exceeds threshold.",
    "probabilistic_acceptance": "Accept arrivals to the longer queue with a decreasing probability as imbalance grows.",
    "hybrid": "Throttle the longer queue and drop arrivals if imbalance becomes too large.",
}
st.sidebar.caption(policy_help[policy])

lambda1 = st.sidebar.number_input("λ₁ (Queue 1 arrival rate)", min_value=0.1, value=3.0, step=0.1)
lambda2 = st.sidebar.number_input("λ₂ (Queue 2 arrival rate)", min_value=0.1, value=3.0, step=0.1)
mu = st.sidebar.number_input("μ (Server rate)", min_value=0.1, value=4.0, step=0.1)

T = st.sidebar.number_input("Simulation Horizon T", min_value=100.0, value=20000.0, step=1000.0)
warmup = st.sidebar.number_input("Warm-up Time", min_value=0.0, value=1000.0, step=100.0)

seed_mode = st.sidebar.selectbox("Randomness", ["Random seed each run", "Fixed seed"], index=0)
seed = None
if seed_mode == "Fixed seed":
    seed = int(st.sidebar.number_input("Seed", min_value=0, value=1, step=1))

record_every = int(st.sidebar.number_input("Record one point every N events", min_value=1, value=50, step=1))

# Defaults
threshold = 3
throttle_rate = 1.0
drop_threshold = 5
acceptance_alpha = 0.2

if policy == "hard_blocking":
    threshold = int(
        st.sidebar.number_input(
            "Imbalance Threshold T",
            min_value=1,
            value=3,
            step=1,
            help="If one queue exceeds the other by more than T, arrivals to the longer queue are blocked.",
        )
    )

elif policy == "rate_throttling":
    threshold = int(
        st.sidebar.number_input(
            "Imbalance Threshold T",
            min_value=1,
            value=3,
            step=1,
            help="If imbalance exceeds T, the longer queue is throttled.",
        )
    )
    throttle_rate = float(
        st.sidebar.number_input(
            "Throttle Rate",
            min_value=0.1,
            value=1.0,
            step=0.1,
            help="Reduced arrival rate applied to the longer queue under throttling.",
        )
    )

elif policy == "probabilistic_acceptance":
    threshold = int(
        st.sidebar.number_input(
            "Imbalance Threshold T",
            min_value=1,
            value=3,
            step=1,
            help="Acceptance probability starts dropping once imbalance exceeds T.",
        )
    )
    acceptance_alpha = float(
        st.sidebar.number_input(
            "Acceptance Slope α",
            min_value=0.01,
            value=0.2,
            step=0.01,
            help="Probability decreases roughly as 1 - α × excess imbalance.",
        )
    )

elif policy == "hybrid":
    threshold = int(
        st.sidebar.number_input(
            "Throttle Threshold T",
            min_value=1,
            value=3,
            step=1,
            help="Throttle the longer queue when imbalance exceeds T.",
        )
    )
    throttle_rate = float(
        st.sidebar.number_input(
            "Throttle Rate",
            min_value=0.1,
            value=1.0,
            step=0.1,
        )
    )
    drop_threshold = int(
        st.sidebar.number_input(
            "Drop Threshold D",
            min_value=1,
            value=5,
            step=1,
            help="Drop arrivals to the longer queue once imbalance exceeds D.",
        )
    )

# -----------------------------
# Diagram + status
# -----------------------------
st.subheader("System Diagram and Status")
colA, colB = st.columns([1.15, 1.0])

with colA:
    fig_diag, ax_diag = plt.subplots(figsize=(9, 4.0))
    ax_diag.set_xlim(0, 12)
    ax_diag.set_ylim(0, 7)
    ax_diag.axis("off")

    queue_color = "#DCEBFF"
    server_color = "#FCE7C8"
    join_color = "#E7F6E7"
    border_color = "#2B2D42"
    arrow_color = "#2B2D42"
    text_color = "#1F2430"

    # Queue 1
    q1_box = patches.FancyBboxPatch(
        (1.1, 4.6), 2.0, 0.9,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=2, edgecolor=border_color, facecolor=queue_color
    )
    ax_diag.add_patch(q1_box)
    ax_diag.text(2.1, 5.05, "Queue 1", ha="center", va="center",
                 fontsize=12, weight="bold", color=text_color)

    # Queue 2
    q2_box = patches.FancyBboxPatch(
        (1.1, 1.7), 2.0, 0.9,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=2, edgecolor=border_color, facecolor=queue_color
    )
    ax_diag.add_patch(q2_box)
    ax_diag.text(2.1, 2.15, "Queue 2", ha="center", va="center",
                 fontsize=12, weight="bold", color=text_color)

    # Internal dividers
    for x in [1.45, 1.8, 2.15]:
        ax_diag.plot([x, x], [4.72, 5.38], color=border_color, lw=1, alpha=0.25)
        ax_diag.plot([x, x], [1.82, 2.48], color=border_color, lw=1, alpha=0.25)

    # Join block
    join_box = patches.FancyBboxPatch(
        (4.8, 3.0), 1.3, 1.1,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=2, edgecolor=border_color, facecolor=join_color
    )
    ax_diag.add_patch(join_box)
    ax_diag.text(5.45, 3.55, "+", ha="center", va="center",
                 fontsize=20, weight="bold", color=text_color)
    ax_diag.text(5.45, 2.65, "Join", ha="center", va="center",
                 fontsize=10, color=text_color)

    # Server
    srv_box = patches.FancyBboxPatch(
        (7.0, 3.05), 2.0, 1.0,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=2, edgecolor=border_color, facecolor=server_color
    )
    ax_diag.add_patch(srv_box)
    ax_diag.text(8.0, 3.55, "Server", ha="center", va="center",
                 fontsize=12, weight="bold", color=text_color)

    # Arrows
    a1 = FancyArrowPatch(
        (3.1, 5.05), (4.8, 3.75),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.9, color=arrow_color
    )
    a2 = FancyArrowPatch(
        (3.1, 2.15), (4.8, 3.35),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.9, color=arrow_color
    )
    a3 = FancyArrowPatch(
        (6.1, 3.55), (7.0, 3.55),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.9, color=arrow_color
    )
    a4 = FancyArrowPatch(
        (9.0, 3.55), (10.8, 3.55),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.9, color=arrow_color
    )

    for a in [a1, a2, a3, a4]:
        ax_diag.add_patch(a)

    # Dynamic labels
    ax_diag.text(1.05, 6.05, f"λ₁ = {lambda1:.2f}", fontsize=13, color=text_color, weight="bold")
    ax_diag.text(1.05, 3.05, f"λ₂ = {lambda2:.2f}", fontsize=13, color=text_color, weight="bold")
    ax_diag.text(7.1, 4.45, f"μ = {mu:.2f}", fontsize=13, color=text_color, weight="bold")

    ax_diag.set_title("Two-Queue Assembly / Synchronization System", fontsize=14, weight="bold", pad=10)
    st.pyplot(fig_diag, use_container_width=True)
    st.caption(f"Current parameters: λ₁ = {lambda1:.2f}, λ₂ = {lambda2:.2f}, μ = {mu:.2f}")

with colB:
    st.markdown("### System Status")

    if abs(lambda1 - lambda2) < 1e-9:
        st.info("Balanced nominal arrival rates: λ₁ ≈ λ₂.")
    elif lambda1 > lambda2:
        st.warning("Nominally, Queue 1 receives arrivals faster than Queue 2.")
    else:
        st.warning("Nominally, Queue 2 receives arrivals faster than Queue 1.")

    if min(lambda1, lambda2) < mu:
        st.success(
            "The server is not the only issue here: synchronization mismatch can dominate the dynamics."
        )
    else:
        st.warning(
            "The server may also become a bottleneck depending on how fast matched pairs are formed."
        )

    st.markdown("### Interpretation")
    if policy in ["none", "hard_blocking"]:
        st.markdown(
            """
- This policy acts directly on the **pre-assembly queues**.
- The key variable is usually the **imbalance** between Queue 1 and Queue 2.
- In assembly systems, one queue often stays near zero while the other grows.
"""
        )
    else:
        st.markdown(
            """
- This policy regulates the **mismatch before pairing**.
- The goal is not just low backlog, but low **imbalance**.
- Better synchronization usually means smoother assembly and less wasted buildup.
"""
        )

run = st.sidebar.button("Run Simulation", use_container_width=True)
animate = st.sidebar.checkbox("Animate sample path", value=False)
compare_all = st.sidebar.checkbox("Compare all policies", value=True)

if run:
    res = simulate_assembly_system(
        policy=policy,
        T=float(T),
        warmup=float(warmup),
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        mu=float(mu),
        seed=seed,
        threshold=int(threshold),
        throttle_rate=float(throttle_rate),
        drop_threshold=int(drop_threshold),
        acceptance_alpha=float(acceptance_alpha),
        record_every=int(record_every),
    )

    st.subheader(f"Selected Policy: {policy}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Queue 1", f"{res.avg_q1:.3f}")
    c2.metric("Average Queue 2", f"{res.avg_q2:.3f}")
    c3.metric("Average Assembly Buffer", f"{res.avg_assembly_buffer:.3f}")
    c4.metric("Average Imbalance", f"{res.avg_imbalance:.3f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Throughput", f"{res.throughput:.3f}")
    c6.metric("Utilization", f"{res.utilization:.3f}")
    c7.metric("Dropped Jobs", f"{res.dropped1 + res.dropped2}")

    st.subheader("Sample Path / Queue Trajectory")

    if animate:
        placeholder = st.empty()
        step = max(5, len(res.times) // 200)

        for k in range(step, len(res.times) + 1, step):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.step(res.times[:k], res.q1_hist[:k], where="post", label="Queue 1")
            ax.step(res.times[:k], res.q2_hist[:k], where="post", label="Queue 2")
            ax.step(res.times[:k], res.assembly_hist[:k], where="post", label="Assembly Buffer")
            ax.set_xlabel("Time")
            ax.set_ylabel("Content")
            ax.set_title(f"Assembly System Trajectory ({policy})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            placeholder.pyplot(fig)
            plt.close(fig)
            pytime.sleep(0.03)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.step(res.times, res.q1_hist, where="post", label="Queue 1")
        ax.step(res.times, res.q2_hist, where="post", label="Queue 2")
        ax.step(res.times, res.assembly_hist, where="post", label="Assembly Buffer")
        ax.set_xlabel("Time")
        ax.set_ylabel("Content")
        ax.set_title(f"Assembly System Trajectory ({policy})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    st.subheader("Diagnostics")

    left, right = st.columns(2)

    with left:
        fig_q, ax_q = plt.subplots(figsize=(6, 4))
        ax_q.bar(
            ["Queue 1", "Queue 2", "Assembly"],
            [res.avg_q1, res.avg_q2, res.avg_assembly_buffer]
        )
        ax_q.set_ylabel("Average Content")
        ax_q.set_title("Average Buffer/Queue Content")
        ax_q.grid(axis="y", alpha=0.3)
        st.pyplot(fig_q, use_container_width=True)

    with right:
        fig_m, ax_m = plt.subplots(figsize=(6, 4))
        names = ["Imbalance", "Throughput", "Utilization"]
        vals = [res.avg_imbalance, res.throughput, res.utilization]
        ax_m.bar(names, vals)
        ax_m.set_title("Other Metrics")
        ax_m.grid(axis="y", alpha=0.3)
        st.pyplot(fig_m, use_container_width=True)

    df_traj = pd.DataFrame(
        {
            "time": res.times,
            "q1": res.q1_hist,
            "q2": res.q2_hist,
            "assembly_buffer": res.assembly_hist,
        }
    )
    csv = df_traj.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download trajectory CSV",
        data=csv,
        file_name=f"assembly_trajectory_{policy}.csv",
        mime="text/csv",
    )

    if compare_all:
        st.subheader("Policy Comparison")

        cmp = compare_policies(
            T=float(T),
            warmup=float(warmup),
            lambda1=float(lambda1),
            lambda2=float(lambda2),
            mu=float(mu),
            seed=seed,
            threshold=int(threshold),
            throttle_rate=float(throttle_rate),
            drop_threshold=int(drop_threshold),
            acceptance_alpha=float(acceptance_alpha),
            record_every=int(record_every),
        )

        df_cmp = pd.DataFrame(cmp)
        st.dataframe(df_cmp, use_container_width=True)

        fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4))
        x = range(len(df_cmp))
        w = 0.35

        ax_cmp.bar([i - w / 2 for i in x], df_cmp["avg_q1"], width=w, label="Avg Q1")
        ax_cmp.bar([i + w / 2 for i in x], df_cmp["avg_q2"], width=w, label="Avg Q2")
        ax_cmp.set_xticks(list(x))
        ax_cmp.set_xticklabels(df_cmp["policy"], rotation=15)
        ax_cmp.set_ylabel("Average Queue Length (log scale)")
        ax_cmp.set_title("Average Queue Length by Policy")
        ax_cmp.set_yscale("log")
        ax_cmp.legend()
        ax_cmp.grid(axis="y", alpha=0.3, which="both")
        st.pyplot(fig_cmp, use_container_width=True)