import torch
import numpy as np
import gymnasium as gym
import chemostat_env
from config import config
import itertools
import csv
import os
import copy

policy_net_file = ""  # Change to your file name
policy_net = torch.load(policy_net_file, map_location=torch.device("cuda"), weights_only=False)
policy_net.eval()


log_file = f"{policy_net_file}_Test.csv"
log_exists = os.path.exists(log_file)
with open(log_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not log_exists:
        writer.writerow(["Action", "Reward", "Done", "State", "Info"])

dt_candidates = config["dt"]
D_candidates = config["D"]
config_combinations = list(itertools.product(
    dt_candidates,
    D_candidates
))
for dt_val, D_val in config_combinations:
    local_config = copy.deepcopy(config)
    local_config["dt"] = dt_val
    local_config["D"] = D_val
    env = gym.make("ChemostatEnv-v0", config=local_config, render=False)
    state, info = env.reset(seed=888)
    done = False
    steps = 0
    total_reward = 0
    while not done:
        device = next(policy_net.parameters()).device
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        action = int(torch.argmax(q_values))

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"Action: {action}, Reward: {reward}")
        steps += 1
        total_reward += reward

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([action, reward, done, state.tolist(), info])

    print(f"Steps: {steps - 1}, Total reward: {total_reward}")

env.close()
