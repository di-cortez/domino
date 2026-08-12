"""Fixed implementation invariants shared by RL training modules."""

# Worker tuning is intentionally a fixed implementation policy rather than an
# experiment axis. Each candidate receives one percent of the reference game
# budget and must improve throughput by at least ten percent to be accepted.
RL_WORKER_AUTOTUNE_FRACTION = 0.01
RL_WORKER_AUTOTUNE_MINIMUM_GAIN = 0.10

# PPO always requests between four and sixteen minibatches. The game-based
# scale and minimum decisions per minibatch remain explicit algorithm controls.
PPO_MIN_MINIBATCHES = 4
PPO_MAX_MINIBATCHES = 16

# A complete PPO buffer may consume at most this fraction of reported free
# VRAM; otherwise the immutable host buffer is streamed by minibatch.
PPO_GPU_BUFFER_SAFETY_FRACTION = 0.70
