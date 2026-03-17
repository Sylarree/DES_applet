from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional

INF = float("inf")


def exp_rv(rate: float, rng: random.Random) -> float:
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
    avg_assembly_buffer: float
    avg_imbalance: float
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
    assembly_hist: List[int]


def simulate_assembly_system(
    policy: str,
    T: float = 20000.0,
    warmup: float = 1000.0,
    lambda1: float = 3.0,
    lambda2: float = 3.0,
    mu: float = 4.0,
    seed: Optional[int] = None,
    threshold: int = 3,
    throttle_rate: float = 1.0,
    drop_threshold: int = 5,
    acceptance_alpha: float = 0.2,
    record_every: int = 50,
) -> SimResult:
    """
    Assembly system:
        - Arrivals join Q1 and Q2 separately
        - When Q1>0 and Q2>0, one item from each can form one assembled job
        - Assembled jobs enter an assembly buffer
        - Server processes assembled jobs at rate mu

    Policies act on the pre-assembly queues and are based on imbalance.

    Policies:
        none:
            no control
        hard_blocking:
            block arrivals to longer queue if imbalance exceeds threshold
        rate_throttling:
            reduce arrival rate of longer queue when imbalance exceeds threshold
        probabilistic_acceptance:
            accept arrivals to longer queue with probability max(0, 1 - alpha * excess)
        hybrid:
            throttle longer queue above threshold; hard drop above drop_threshold
    """

    rng = random.Random(seed)

    # State
    t = 0.0
    q1 = 0
    q2 = 0
    assembly_buffer = 0
    server_busy = False

    # Event calendar
    tA1 = exp_rv(lambda1, rng)
    tA2 = exp_rv(lambda2, rng)
    tD = INF

    # Stats
    arrivals1 = 0
    arrivals2 = 0
    dropped1 = 0
    dropped2 = 0
    completed = 0

    area_q1 = 0.0
    area_q2 = 0.0
    area_assembly = 0.0
    area_busy = 0.0
    area_imbalance = 0.0
    last_t = 0.0

    # History
    times: List[float] = [0.0]
    q1_hist: List[int] = [0]
    q2_hist: List[int] = [0]
    assembly_hist: List[int] = [0]
    event_counter = 0

    def maybe_record(now: float) -> None:
        nonlocal event_counter
        event_counter += 1
        if record_every <= 1 or event_counter % record_every == 0:
            times.append(now)
            q1_hist.append(q1)
            q2_hist.append(q2)
            assembly_hist.append(assembly_buffer)

    def update_time_averages(new_t: float) -> None:
        nonlocal area_q1, area_q2, area_assembly, area_busy, area_imbalance, last_t
        start = max(last_t, warmup)
        end = max(min(new_t, T), warmup)

        if end > start:
            dt = end - start
            area_q1 += q1 * dt
            area_q2 += q2 * dt
            area_assembly += assembly_buffer * dt
            area_busy += (1.0 if server_busy else 0.0) * dt
            area_imbalance += abs(q1 - q2) * dt

        last_t = new_t

    def imbalance() -> int:
        return q1 - q2

    def try_assemble_and_start_service(now: float) -> None:
        """
        Form as many pairs as possible from Q1 and Q2 into assembly buffer.
        Then start service if server is idle and buffer has jobs.
        """
        nonlocal q1, q2, assembly_buffer, server_busy, tD

        pairs = min(q1, q2)
        if pairs > 0:
            q1 -= pairs
            q2 -= pairs
            assembly_buffer += pairs

        if (not server_busy) and assembly_buffer > 0:
            assembly_buffer -= 1
            server_busy = True
            tD = now + exp_rv(mu, rng)
        elif (not server_busy) and assembly_buffer == 0:
            tD = INF

    def arrival_allowed(queue_idx: int) -> bool:
        """
        Decide whether arrival is admitted based on current imbalance and policy.
        The longer queue is the one potentially controlled.
        """
        nonlocal dropped1, dropped2

        d = imbalance()

        longer_q1 = d > threshold
        longer_q2 = -d > threshold

        if policy == "none":
            return True

        if policy == "hard_blocking":
            if queue_idx == 1 and longer_q1:
                dropped1 += 1
                return False
            if queue_idx == 2 and longer_q2:
                dropped2 += 1
                return False
            return True

        if policy == "probabilistic_acceptance":
            if queue_idx == 1 and d > threshold:
                excess = d - threshold
                p = max(0.0, 1.0 - acceptance_alpha * excess)
                ok = rng.random() < p
                if not ok:
                    dropped1 += 1
                return ok
            if queue_idx == 2 and -d > threshold:
                excess = -d - threshold
                p = max(0.0, 1.0 - acceptance_alpha * excess)
                ok = rng.random() < p
                if not ok:
                    dropped2 += 1
                return ok
            return True

        if policy == "hybrid":
            if queue_idx == 1:
                if d >= drop_threshold:
                    dropped1 += 1
                    return False
            if queue_idx == 2:
                if -d >= drop_threshold:
                    dropped2 += 1
                    return False
            return True

        # rate_throttling does not drop arrivals
        return True

    def next_arrival_time(now: float, queue_idx: int) -> float:
        """
        For throttling / hybrid, if one queue is much longer than the other,
        reduce that queue's arrival rate.
        """
        base_rate = lambda1 if queue_idx == 1 else lambda2
        d = imbalance()

        if policy in ("rate_throttling", "hybrid"):
            if queue_idx == 1 and d > threshold:
                return now + exp_rv(throttle_rate, rng)
            if queue_idx == 2 and -d > threshold:
                return now + exp_rv(throttle_rate, rng)

        return now + exp_rv(base_rate, rng)

    # Main DES loop
    while True:
        t_next = min(tA1, tA2, tD, T)
        update_time_averages(t_next)
        t = t_next

        if t >= T:
            break

        # Arrival to Q1
        if t == tA1:
            arrivals1 += 1
            if arrival_allowed(1):
                q1 += 1
                try_assemble_and_start_service(t)
            tA1 = next_arrival_time(t, 1)
            maybe_record(t)

        # Arrival to Q2
        elif t == tA2:
            arrivals2 += 1
            if arrival_allowed(2):
                q2 += 1
                try_assemble_and_start_service(t)
            tA2 = next_arrival_time(t, 2)
            maybe_record(t)

        # Departure
        else:
            completed += 1
            server_busy = False
            try_assemble_and_start_service(t)
            maybe_record(t)

    if times[-1] != t:
        times.append(t)
        q1_hist.append(q1)
        q2_hist.append(q2)
        assembly_hist.append(assembly_buffer)

    observed_time = max(T - warmup, 1e-12)

    return SimResult(
        policy=policy,
        avg_q1=area_q1 / observed_time,
        avg_q2=area_q2 / observed_time,
        avg_assembly_buffer=area_assembly / observed_time,
        avg_imbalance=area_imbalance / observed_time,
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
        assembly_hist=assembly_hist,
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
    drop_threshold: int = 5,
    acceptance_alpha: float = 0.2,
    record_every: int = 50,
):
    policies = [
        "none",
        "hard_blocking",
        "rate_throttling",
        "probabilistic_acceptance",
        "hybrid",
    ]

    rows = []
    for pol in policies:
        res = simulate_assembly_system(
            policy=pol,
            T=T,
            warmup=warmup,
            lambda1=lambda1,
            lambda2=lambda2,
            mu=mu,
            seed=seed,
            threshold=threshold,
            throttle_rate=throttle_rate,
            drop_threshold=drop_threshold,
            acceptance_alpha=acceptance_alpha,
            record_every=record_every,
        )
        rows.append(
            {
                "policy": pol,
                "avg_q1": res.avg_q1,
                "avg_q2": res.avg_q2,
                "avg_assembly_buffer": res.avg_assembly_buffer,
                "avg_imbalance": res.avg_imbalance,
                "throughput": res.throughput,
                "utilization": res.utilization,
                "drops": res.dropped1 + res.dropped2,
            }
        )
    return rows