import numpy as np
import matplotlib.pyplot as plt

# --- Moving average smoothing ---
def smooth(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


# --- Load raw returns ---
lane_raw = np.load("results/MORL_lane.npy", allow_pickle=True)
safe_raw = np.load("results/MORL_safe.npy", allow_pickle=True)
fast_raw = np.load("results/MORL_fast.npy", allow_pickle=True)

# --- Smooth curves ---
WINDOW = 10   # change to 20 for even smoother curves

lane = smooth(lane_raw, WINDOW)
safe = smooth(safe_raw, WINDOW)
fast = smooth(fast_raw, WINDOW)

plt.figure(figsize=(10, 6))

plt.plot(lane, label=f"Lane-Keeping Policy (smooth={WINDOW})", linewidth=2)
plt.plot(safe, label=f"Safety Policy (smooth={WINDOW})", linewidth=2)
plt.plot(fast, label=f"Speed Policy (smooth={WINDOW})", linewidth=2)

plt.title("Multi-Objective RL — Smoothed Reward Comparison")
plt.xlabel("Episodes")
plt.ylabel("Total Return")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/MORL_comparison_smooth.png", dpi=300)
plt.show()
