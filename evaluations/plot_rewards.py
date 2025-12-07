import os
from typing import List

import matplotlib.pyplot as plt


def plot_rewards(rewards: List[float], out_path: str, title: str = "Training Returns"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
