import numpy as np
import matplotlib.pyplot as plt

def smooth(x, w=10):
    return np.convolve(x, np.ones(w)/w, mode="valid")

# Load curves
dqn = np.load("results/DQN.npy")
dqn_exp = np.load("results/DQN_exp.npy")
noisy = np.load("results/NoisyDQN.npy")

plt.figure(figsize=(10,6))

plt.plot(smooth(dqn), label="DQN (Linear ε-decay)", linewidth=2)
plt.plot(smooth(dqn_exp), label="DQN (Exponential ε-decay)", linewidth=2)
plt.plot(smooth(noisy), label="NoisyNet Exploration", linewidth=2)

plt.title("Exploration Strategy Comparison")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid(True)
plt.legend()

plt.savefig("results/plots/Exploration_Comparison.png")
plt.show()
