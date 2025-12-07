import gymnasium as gym
import highway_env 
from envs.multi_objective_wrapper import MultiObjectiveHighway

def make_env(name, config):
    env = gym.make(name)

    # Apply MORL wrapper:
    if "weights" in config:
        print(">> MultiObjective wrapper active with weights:", config["weights"])
        env = MultiObjectiveHighway(env, config["weights"])

    # Configure environment to output individual reward components
    env.unwrapped.configure({
        "observation": {"type": "Kinematics"},
        "duration": 40,
        "simulation_frequency": 15,
        "policy_frequency": 5,

        "reward_speed_range": [20, 30],

        # enable component-level rewards
        "reward_speed": 1.0,
        "reward_lane_centering": 1.0,
        "reward_collision": -5.0,
        "reward_headway": 1.0,

        "collision_reward": -5.0,
        "off_road_penalty": -5.0,
    })

    return env
