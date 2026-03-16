import io
import time as pytime
import math
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from simulator import simulate_two_queue_system, compare_policies

st.set_page_config(page_title="Two-Queue Control Applet", layout="wide")

st.title("Two-Queue Merge System Control Applet")
st.caption(
    "Interactive DES simulator for a two-queue / one-server merge system with multiple control policies."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Model Parameters")

policy = st.sidebar.selectbox(
    "Control Policy",
    ["none", "priority", "threshold_priority", "rate_throttling", "hybrid"],
    index=0,
)

policy_help = {
    "none": "Baseline: random tie-breaking, no admission control.",
    "priority": "Serve the longer queue first.",
    "threshold_priority": "Prioritize a queue once it exceeds a threshold.",
    "rate_throttling": "Reduce arrival rate when a queue becomes congested.",
    "hybrid": "Threshold-based priority + throttling + dropping above a limit.",
}
st.sidebar.caption(policy_help[policy])

lambda1 = st.sidebar.number_input("λ₁ (Queue 1 arrival rate)", min_value=0.1, value=3.0, step=0.1)
lambda2 = st.sidebar.number_input("λ₂ (Queue 2 arrival rate)", min_value=0.1, value=3.0, step=0.1)
mu = st.sidebar.number_input("μ (Service rate)", min_value=0.1, value=4.0, step=0.1)

T = st.sidebar.number_input("Simulation Horizon T", min_value=100.0, value=20000.0, step=1000.0)
warmup = st.sidebar.number_input("Warm-up Time", min_value=0.0, value=1000.0, step=100.0)
seed_mode = st.sidebar.selectbox("Randomness", ["Random seed each run", "Fixed seed"], index=0)

seed = None
if seed_mode == "Fixed seed":
    seed = int(st.sidebar.number_input("Seed", min_value=0, value=1, step=1))

record_every = int(
    st.sidebar.number_input("Record one point every N events", min_value=1, value=50, step=1)
)

# -----------------------------
# Policy-specific parameters
# Editable + default values
# -----------------------------
threshold = 3
throttle_rate = 1.0
hybrid_drop_threshold = 5

if policy == "threshold_priority":
    threshold = int(
        st.sidebar.number_input(
            "Threshold T",
            min_value=1,
            value=3,
            step=1,
            help="Queue is prioritized when its content reaches/exceeds this threshold.",
        )
    )

elif policy == "rate_throttling":
    threshold = int(
        st.sidebar.number_input(
            "Congestion Threshold T",
            min_value=1,
            value=3,
            step=1,
            help="When a queue reaches/exceeds this threshold, its arrival rate is throttled.",
        )
    )
    throttle_rate = float(
        st.sidebar.number_input(
            "Throttle Rate",
            min_value=0.1,
            value=1.0,
            step=0.1,
            help="Reduced arrival rate applied once throttling is active.",
        )
    )

elif policy == "hybrid":
    threshold = int(
        st.sidebar.number_input(
            "Priority / Throttle Threshold T",
            min_value=1,
            value=3,
            step=1,
            help="Threshold that triggers stronger control behavior.",
        )
    )
    throttle_rate = float(
        st.sidebar.number_input(
            "Throttle Rate",
            min_value=0.1,
            value=1.0,
            step=0.1,
            help="Reduced arrival rate when throttling is active.",
        )
    )
    hybrid_drop_threshold = int(
        st.sidebar.number_input(
            "Drop Threshold D",
            min_value=1,
            value=5,
            step=1,
            help="Arrivals are dropped when queue length reaches/exceeds this value.",
        )
    )

# -----------------------------
# System load indicator
# -----------------------------
rho_nominal = (lambda1 + lambda2) / mu

st.subheader("System Diagram and Load Status")
colA, colB = st.columns([1.2, 1.0])

with colA:
    fig_diag, ax_diag = plt.subplots(figsize=(9, 3.8))
    ax_diag.set_xlim(0, 12)
    ax_diag.set_ylim(0, 7)
    ax_diag.axis("off")

    queue_color = "#DCEBFF"
    server_color = "#FFE7C7"
    border_color = "#2B2D42"
    arrow_color = "#222222"
    text_color = "#1F2430"

    # Queue 1
    q1_box = plt.matplotlib.patches.FancyBboxPatch(
        (1.0, 4.5), 2.4, 1.0,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=2, edgecolor=border_color, facecolor=queue_color
    )
    ax_diag.add_patch(q1_box)
    ax_diag.text(2.2, 5.0, "Queue 1", ha="center", va="center",
                 fontsize=12, weight="bold", color=text_color)

    # Queue 2
    q2_box = plt.matplotlib.patches.FancyBboxPatch(
        (1.0, 1.7), 2.4, 1.0,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=2, edgecolor=border_color, facecolor=queue_color
    )
    ax_diag.add_patch(q2_box)
    ax_diag.text(2.2, 2.2, "Queue 2", ha="center", va="center",
                 fontsize=12, weight="bold", color=text_color)

    # Optional internal queue dividers
    for x in [1.45, 1.9, 2.35]:
        ax_diag.plot([x, x], [4.58, 5.42], color=border_color, lw=1, alpha=0.35)
        ax_diag.plot([x, x], [1.78, 2.62], color=border_color, lw=1, alpha=0.35)

    # Server
    srv_x, srv_y, srv_w, srv_h = 6.3, 3.0, 2.0, 1.3
    srv_box = plt.matplotlib.patches.FancyBboxPatch(
        (srv_x, srv_y), srv_w, srv_h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=2, edgecolor=border_color, facecolor=server_color
    )
    ax_diag.add_patch(srv_box)
    ax_diag.text(srv_x + srv_w/2, srv_y + srv_h/2, "Server",
                 ha="center", va="center", fontsize=12, weight="bold", color=text_color)

    # Horizontal arrows from queues
    ax_diag.annotate("", xy=(5.0, 5.0), xytext=(3.4, 5.0),
                     arrowprops=dict(arrowstyle="->", lw=2.0, color=arrow_color))
    ax_diag.annotate("", xy=(5.0, 2.2), xytext=(3.4, 2.2),
                     arrowprops=dict(arrowstyle="->", lw=2.0, color=arrow_color))

    # Clean merge connector (stop before server)
    ax_diag.plot([5.0, 5.7], [5.0, 4.2], color=arrow_color, lw=1.8)
    ax_diag.plot([5.0, 5.7], [2.2, 3.1], color=arrow_color, lw=1.8)

    # Short arrow into server
    ax_diag.annotate("", xy=(srv_x, 3.65), xytext=(5.7, 3.65),
                     arrowprops=dict(arrowstyle="->", lw=2.0, color=arrow_color))

    # Output arrow
    ax_diag.annotate("", xy=(10.6, 3.65), xytext=(srv_x + srv_w, 3.65),
                     arrowprops=dict(arrowstyle="->", lw=2.0, color=arrow_color))

    # Labels
    ax_diag.text(1.0, 6.0, rf"$\lambda_1 = {lambda1:.2f}$", fontsize=12, color=text_color)
    ax_diag.text(1.0, 3.15, rf"$\lambda_2 = {lambda2:.2f}$", fontsize=12, color=text_color)
    ax_diag.text(6.55, 4.75, rf"$\mu = {mu:.2f}$", fontsize=12, color=text_color)

    ax_diag.set_title("Two-Queue / One-Server Merge System", fontsize=14, weight="bold", pad=10)
    st.pyplot(fig_diag, use_container_width=True)

with colB:
    st.markdown("### System Status")

    if rho_nominal < 1:
        st.success(
            f"Nominal load ρ = (λ₁ + λ₂)/μ = {rho_nominal:.2f} → nominally stable"
        )
    else:
        if policy in ["none", "priority", "threshold_priority"]:
            st.error(
                f"Nominal load ρ = (λ₁ + λ₂)/μ = {rho_nominal:.2f} → overloaded, and this policy does not regulate input, so instability is expected."
            )
        else:
            st.warning(
                f"Nominal load ρ = (λ₁ + λ₂)/μ = {rho_nominal:.2f} → nominally overloaded, but this controller may still stabilize the system by reducing effective input."
            )

    st.markdown("### Interpretation")
    if policy in ["none", "priority", "threshold_priority"]:
        st.markdown(
            r"""
    - This policy mainly changes **service order**.
    - If nominal load exceeds capacity, backlog should still grow.
    - So here the main effect is **who waits more**, not whether the system is stable.
    """
        )
    else:
        st.markdown(
            r"""
    - This policy acts on the **input side**.
    - It can reduce the effective arrival load seen by the server.
    - That is why bounded queue trajectories may appear even when nominal \(\rho > 1\).
    """
        )

run = st.sidebar.button("Run Simulation", use_container_width=True)
animate = st.sidebar.checkbox("Animate sample path", value=False)
compare_all = st.sidebar.checkbox("Compare all policies", value=True)

# -----------------------------
# Run one selected policy
# -----------------------------
if run:
    res = simulate_two_queue_system(
        policy=policy,
        T=float(T),
        warmup=float(warmup),
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        mu=float(mu),
        seed=seed,
        threshold=int(threshold),
        throttle_rate=float(throttle_rate),
        hybrid_drop_threshold=int(hybrid_drop_threshold),
        record_every=int(record_every),
    )

    st.subheader(f"Selected Policy: {policy}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Queue 1", f"{res.avg_q1:.3f}")
    c2.metric("Average Queue 2", f"{res.avg_q2:.3f}")
    c3.metric("Average Total Queue", f"{res.avg_total:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Throughput", f"{res.throughput:.3f}")
    c5.metric("Utilization", f"{res.utilization:.3f}")
    c6.metric("Dropped Jobs", f"{res.dropped1 + res.dropped2}")

    # Queue trajectory
    st.subheader("Sample Path / Queue Trajectory")

    if animate:
        placeholder = st.empty()
        step = max(5, len(res.times) // 200)

        for k in range(step, len(res.times) + 1, step):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.step(res.times[:k], res.q1_hist[:k], where="post", label="Queue 1")
            ax.step(res.times[:k], res.q2_hist[:k], where="post", label="Queue 2")
            ax.set_xlabel("Time")
            ax.set_ylabel("Queue Length")
            ax.set_title(f"Queue Evolution ({policy})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            placeholder.pyplot(fig)
            plt.close(fig)
            pytime.sleep(0.03)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.step(res.times, res.q1_hist, where="post", label="Queue 1")
        ax.step(res.times, res.q2_hist, where="post", label="Queue 2")
        ax.set_xlabel("Time")
        ax.set_ylabel("Queue Length")
        ax.set_title(f"Queue Evolution ({policy})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    # Diagnostics
    st.subheader("Diagnostics")

    left, right = st.columns(2)

    with left:
        fig_q, ax_q = plt.subplots(figsize=(6, 4))
        ax_q.bar(["Queue 1", "Queue 2"], [res.avg_q1, res.avg_q2])
        ax_q.set_ylabel("Average Queue Length")
        ax_q.set_title("Average Queue Lengths")
        ax_q.grid(axis="y", alpha=0.3)
        st.pyplot(fig_q, use_container_width=True)

    with right:
        fig_m, ax_m = plt.subplots(figsize=(6, 4))
        names = ["Throughput", "Utilization"]
        vals = [res.throughput, res.utilization]
        ax_m.bar(names, vals)
        ax_m.set_title("Other Metrics")
        ax_m.grid(axis="y", alpha=0.3)
        st.pyplot(fig_m, use_container_width=True)

    # Trajectory download
    df_traj = pd.DataFrame(
        {"time": res.times, "q1": res.q1_hist, "q2": res.q2_hist}
    )
    csv = df_traj.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download trajectory CSV",
        data=csv,
        file_name=f"trajectory_{policy}.csv",
        mime="text/csv",
    )

    # -----------------------------
    # Compare all policies
    # -----------------------------
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
            hybrid_drop_threshold=int(hybrid_drop_threshold),
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
        ax_cmp.set_yscale("log")   # <- important
        ax_cmp.legend()
        ax_cmp.grid(axis="y", alpha=0.3, which="both")

        st.pyplot(fig_cmp, use_container_width=True)