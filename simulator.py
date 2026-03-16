from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional

INF = float("inf")


def exp_rv(rate: float, rng: random.Random) -> float:
    """Generate an exponential random variable with the given rate."""
    if rate <= 0:
        raise ValueError("Rate must be positive.")
    u = rng.random()
    while u <= 0.0:
        u = rng.random()
    return -math.log(u) / rate


@dataclass
class SimResult:
    policy: str
    avg_q1: float
    avg_q2: float
    avg_total: float
    throughput: float
    utilization: float
    dropped1: int
    dropped2: int
    arrivals1: int
    arrivals2: int
    completed: int
    sim_time: float
    times: List[float]
    q1_hist: List[int]
    q2_hist: List[int]


def simulate_two_queue_system(
    policy: str,
    T: float = 20000.0,
    warmup: float = 1000.0,
    lambda1: float = 3.0,
    lambda2: float = 3.0,
    mu: float = 4.0,
    seed: Optional[int] = None,
    threshold: int = 3,
    throttle_rate: float = 1.0,
    hybrid_drop_threshold: int = 5,
    record_every: int = 50,
) -> SimResult:
    """
    Simulate a two-queue / one-server merge system.

    Policies:
        none:
            random tie-breaking, no input control
        priority:
            serve the longer queue first (tie -> Q1)
        threshold_priority:
            if a queue reaches/exceeds threshold, prioritize it
        rate_throttling:
            if Qi >= threshold, reduce that queue's arrival rate to throttle_rate
        hybrid:
            threshold-priority service + throttling + hard drop if Qi >= hybrid_drop_threshold
    """
    rng = random.Random(seed)

    # State
    t = 0.0
    q1 = 0
    q2 = 0
    server_busy = False

    # Event calendar
    tA1 = exp_rv(lambda1, rng)
    tA2 = exp_rv(lambda2, rng)
    tD = INF

    # Statistics
    arrivals1 = 0
    arrivals2 = 0
    dropped1 = 0
    dropped2 = 0
    completed = 0

    area_q1 = 0.0
    area_q2 = 0.0
    area_busy = 0.0
    last_t = 0.0

    # History for plotting
    times: List[float] = [0.0]
    q1_hist: List[int] = [0]
    q2_hist: List[int] = [0]
    event_counter = 0

    def maybe_record(now: float) -> None:
        nonlocal event_counter
        event_counter += 1
        if record_every <= 1 or event_counter % record_every == 0:
            times.append(now)
            q1_hist.append(q1)
            q2_hist.append(q2)

    def update_time_averages(new_t: float) -> None:
        nonlocal area_q1, area_q2, area_busy, last_t
        start = max(last_t, warmup)
        end = max(min(new_t, T), warmup)

        if end > start:
            dt = end - start
            area_q1 += q1 * dt
            area_q2 += q2 * dt
            area_busy += (1.0 if server_busy else 0.0) * dt

        last_t = new_t

    def choose_next_queue() -> Optional[int]:
        """Choose which queue to serve next."""
        if q1 + q2 == 0:
            return None

        if policy == "none":
            if q1 > 0 and q2 > 0:
                return 1 if rng.random() < 0.5 else 2
            if q1 > 0:
                return 1
            return 2

        if policy in ("threshold_priority", "hybrid"):
            if q1 >= threshold and q2 >= threshold:
                return 1 if q1 >= q2 else 2
            if q1 >= threshold:
                return 1
            if q2 >= threshold:
                return 2

        # default behavior for priority / rate_throttling / tie-breaks
        if q1 > q2:
            return 1
        if q2 > q1:
            return 2
        return 1

    def maybe_start_service(now: float) -> None:
        nonlocal server_busy, q1, q2, tD

        if server_busy:
            return

        nxt = choose_next_queue()
        if nxt is None:
            tD = INF
            return

        server_busy = True
        if nxt == 1:
            q1 -= 1
        else:
            q2 -= 1

        tD = now + exp_rv(mu, rng)

    def next_arrival_time(now: float, queue_idx: int) -> float:
        """Schedule next arrival according to current policy and congestion state."""
        base_rate = lambda1 if queue_idx == 1 else lambda2

        if policy in ("rate_throttling", "hybrid"):
            if queue_idx == 1 and q1 >= threshold:
                return now + exp_rv(throttle_rate, rng)
            if queue_idx == 2 and q2 >= threshold:
                return now + exp_rv(throttle_rate, rng)

        return now + exp_rv(base_rate, rng)

    # Main DES loop
    while True:
        t_next = min(tA1, tA2, tD, T)
        update_time_averages(t_next)
        t = t_next

        if t >= T:
            break

        # Arrival to Queue 1
        if t == tA1:
            arrivals1 += 1
            admit = True

            if policy == "hybrid" and q1 >= hybrid_drop_threshold:
                admit = False
                dropped1 += 1

            if admit:
                q1 += 1
                maybe_start_service(t)

            tA1 = next_arrival_time(t, 1)
            maybe_record(t)

        # Arrival to Queue 2
        elif t == tA2:
            arrivals2 += 1
            admit = True

            if policy == "hybrid" and q2 >= hybrid_drop_threshold:
                admit = False
                dropped2 += 1

            if admit:
                q2 += 1
                maybe_start_service(t)

            tA2 = next_arrival_time(t, 2)
            maybe_record(t)

        # Departure
        else:
            completed += 1
            server_busy = False
            maybe_start_service(t)
            maybe_record(t)

    if times[-1] != t:
        times.append(t)
        q1_hist.append(q1)
        q2_hist.append(q2)

    observed_time = max(T - warmup, 1e-12)

    return SimResult(
        policy=policy,
        avg_q1=area_q1 / observed_time,
        avg_q2=area_q2 / observed_time,
        avg_total=(area_q1 + area_q2) / observed_time,
        throughput=completed / observed_time,
        utilization=area_busy / observed_time,
        dropped1=dropped1,
        dropped2=dropped2,
        arrivals1=arrivals1,
        arrivals2=arrivals2,
        completed=completed,
        sim_time=T,
        times=times,
        q1_hist=q1_hist,
        q2_hist=q2_hist,
    )


def compare_policies(
    T: float = 20000.0,
    warmup: float = 1000.0,
    lambda1: float = 3.0,
    lambda2: float = 3.0,
    mu: float = 4.0,
    seed: Optional[int] = None,
    threshold: int = 3,
    throttle_rate: float = 1.0,
    hybrid_drop_threshold: int = 5,
    record_every: int = 50,
):
    policies = ["none", "priority", "threshold_priority", "rate_throttling", "hybrid"]
    rows = []

    for pol in policies:
        res = simulate_two_queue_system(
            policy=pol,
            T=T,
            warmup=warmup,
            lambda1=lambda1,
            lambda2=lambda2,
            mu=mu,
            seed=seed,
            threshold=threshold,
            throttle_rate=throttle_rate,
            hybrid_drop_threshold=hybrid_drop_threshold,
            record_every=record_every,
        )

        rows.append(
            {
                "policy": pol,
                "avg_q1": res.avg_q1,
                "avg_q2": res.avg_q2,
                "avg_total": res.avg_total,
                "throughput": res.throughput,
                "utilization": res.utilization,
                "drops": res.dropped1 + res.dropped2,
            }
        )

    return rows