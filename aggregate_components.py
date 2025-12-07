import numpy as np

def load(path):
    data = np.load(path, allow_pickle=True)
    avg = {k: np.mean([ep[k] for ep in data]) for k in data[0].keys()}
    return avg

fast = load("results/MORL_fast_components.npy")
lane = load("results/MORL_lane_components.npy")
safe = load("results/MORL_safe_components.npy")

summary = {"fast": fast, "lane": lane, "safe": safe}

np.save("results/MORL_summary.npy", summary)

print(summary)
