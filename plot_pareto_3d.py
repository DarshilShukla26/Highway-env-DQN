import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

summary = np.load("results/MORL_summary.npy", allow_pickle=True).item()

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

for name, comp in summary.items():
    ax.scatter(comp["speed"], comp["lane"], comp["collision"], s=120)
    ax.text(comp["speed"], comp["lane"], comp["collision"], name.upper())

ax.set_xlabel("Speed")
ax.set_ylabel("Lane Keeping")
ax.set_zlabel("Safety")
plt.title("3D Pareto Frontier")
plt.savefig("results/pareto_3d.png", dpi=300)
plt.show()
