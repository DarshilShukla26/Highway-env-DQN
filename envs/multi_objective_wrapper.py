import gymnasium as gym
import highway_env 
class MultiObjectiveHighway(gym.Wrapper):
    def __init__(self, env, weights):
        super().__init__(env)
        self.weights = weights

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract reward components from info (default 0.0)
        components = {
            "speed": info.get("reward_speed", 0.0),
            "lane": info.get("reward_lane_centering", 0.0),
            "collision": info.get("reward_collision", 0.0),
            "headway": info.get("reward_headway", 0.0),
        }

        # Weighted scalar
        multi_r = (
            self.weights["speed"] * components["speed"] +
            self.weights["lane"] * components["lane"] +
            self.weights["collision"] * components["collision"] +
            self.weights["headway"] * components["headway"]
        )

        # store component logs
        info["multi_reward_components"] = components

        return obs, multi_r, terminated, truncated, info
