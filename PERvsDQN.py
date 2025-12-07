import numpy as np
import matplotlib.pyplot as plt

# Load reward curves
dqn_rewards = np.load("results/DQN.npy")
per_rewards = np.load("results/PERDQN.npy")

plt.figure(figsize=(10, 6))

def smooth(x, w=10):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.plot(smooth(dqn_rewards), label="DQN (smoothed)", linewidth=2)
plt.plot(smooth(per_rewards), label="PERDQN (smoothed)", linewidth=2)


plt.title("PER vs Vanilla DQN — Sample Efficiency Comparison")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid(True)
plt.legend()

# Save figure
plt.savefig("results/plots/PER_vs_DQN.png")
plt.show()
