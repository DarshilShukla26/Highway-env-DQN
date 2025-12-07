import numpy as np
import matplotlib.pyplot as plt

dqn = np.load("results/DQN.npy")
ddqn = np.load("results/DoubleDQN.npy")

plt.figure(figsize=(10,6))
plt.plot(dqn, label="DQN")
plt.plot(ddqn, label="DoubleDQN")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("DQN vs DoubleDQN Training Curve")
plt.grid()
plt.show()
