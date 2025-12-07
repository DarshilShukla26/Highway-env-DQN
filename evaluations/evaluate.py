from typing import List

import numpy as np


def evaluate(env, agent, episodes: int = 10) -> List[float]:
    scores = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.act_greedy(state)
            state, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r
        scores.append(total)
    return scores
