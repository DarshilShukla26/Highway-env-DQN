import matplotlib.pyplot as plt
import numpy as np

summary = np.load("results/MORL_summary.npy", allow_pickle=True).item()

points = {}
for name, comp in summary.items():
    points[name] = (comp["speed"], comp["collision"])

plt.figure(figsize=(7,6))

for name, (s, c) in points.items():
    plt.scatter(s, c, s=160)
    plt.text(s+0.02, c+0.02, name.upper(), fontsize=12)

plt.xlabel("Average Speed Reward")
plt.ylabel("Average Collision (Safety) Reward")
plt.title("Pareto Frontier — Speed vs Safety")
plt.grid(True)
plt.savefig("results/pareto_2d.png", dpi=300)
plt.show()
