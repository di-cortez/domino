"""Fixed implementation invariants shared by RL training modules."""

# Worker tuning is intentionally a fixed implementation policy rather than an
# experiment axis. Each candidate receives one percent of the reference game
# budget and must improve throughput by at least ten percent to be accepted.
RL_WORKER_AUTOTUNE_FRACTION = 0.01
RL_WORKER_AUTOTUNE_MINIMUM_GAIN = 0.10
