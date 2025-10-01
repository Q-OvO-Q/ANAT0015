import chemostat_env
import gymnasium as gym
from Agent import DeepQAgent
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv
import os
import torch
import datetime
import itertools
from config import config
import copy


# Fixed Hyperparameters
n_episodes = 
epsilon_start = 
epsilon_end = 
buffer_size = 
batch_size = 
tau = 
model = 'DQN'
logging = False


hyper_grid = {
    "learning_rate": [],
    "gamma": [],
    "epsilon_decay": []
}


dt_candidates = config["dt"]
D_candidates = config["D"]


hyper_combinations = list(itertools.product(
    hyper_grid["learning_rate"],
    hyper_grid["gamma"],
    hyper_grid["epsilon_decay"]
))
config_combinations = list(itertools.product(
    dt_candidates,
    D_candidates
))


all_rewards = {}
all_lengths = {}
all_losses = {}
all_q_values = {}


results_summary = []


for lr, gamma_val, eps_decay in hyper_combinations:
    for dt_val, D_val in config_combinations:
        # Update local config for different combinations
        local_config = copy.deepcopy(config)
        local_config["dt"] = dt_val
        local_config["D"] = D_val

        print(
            f"\nlearning_rate={lr}, gamma={gamma_val}, epsilon_decay={eps_decay}, dt={dt_val}, D={D_val}")

        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        env = gym.make("ChemostatEnv-v0", config=local_config)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = DeepQAgent(
            env,
            state_dim,
            action_dim,
            learning_rate=lr,
            gamma=gamma_val,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=eps_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            model=model
        )

        if logging:
            log_file = f"{formatted_time}_lr{lr}_gamma{gamma_val}_epsdecay{eps_decay}_dt{dt_val}_D{D_val}.csv"
            log_exists = os.path.exists(log_file)
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not log_exists:
                    writer.writerow(["Episode", "Action", "Reward", "Done", "State", "Info"])

        episode_rewards = []
        episode_lengths = []
        episode_losses = []
        episode_avg_qs = []

        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

        for episode in tqdm(range(n_episodes)):
            state, info = env.reset(seed=888)
            done = False
            episode_reward = 0
            ep_loss_values = []
            ep_q_values = []

            while not done:
                obs = state
                action = agent.get_action(obs)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_obs = next_state
                agent.store_transition(obs, action, reward, next_obs, done)
                result = agent.update()
                if result is not None:
                    loss_val, q_val = result
                    if loss_val is not None:
                        ep_loss_values.append(loss_val)
                        ep_q_values.append(q_val)

                if logging:
                    with open(log_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([episode, action, reward, done, obs.tolist(), info])

                state = next_state
                episode_reward += reward

            episode_rewards.append(episode_reward)
            episode_lengths.append(env.length_queue[-1] if hasattr(env, "length_queue") else 0)
            if ep_loss_values:
                episode_losses.append(np.mean(ep_loss_values))
                episode_avg_qs.append(np.mean(ep_q_values))
            else:
                episode_losses.append(0)
                episode_avg_qs.append(0)

            print(f"Episode {episode}, Reward: {episode_reward}, EPS: {agent.epsilon}")

        env.close()
        # Save
        torch.save(agent.target_net,
                   f"{formatted_time}_target_net.pt")
        torch.save(agent.policy_net,
                   f"{formatted_time}_policy_net.pt")


        key = f"lr={lr},γ={gamma_val},decay={eps_decay},dt={dt_val},D={D_val}"
        all_rewards[key] = episode_rewards
        all_lengths[key] = episode_lengths
        all_losses[key] = episode_losses
        all_q_values[key] = episode_avg_qs


        results_summary.append({
            "learning_rate": lr,
            "gamma": gamma_val,
            "epsilon_decay": eps_decay,
            "dt": dt_val,
            "D": D_val,
            "model": model,
            "truncation": config["truncation"],
            "guided": config["guided"]
        })


rolling_length = 25

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

# 定义一个辅助函数计算滚动平均
def moving_average(data, window, mode="valid"):
    return np.convolve(np.array(data), np.ones(window), mode=mode) / window

# 绘制奖励的滚动平均曲线
for key, rewards in all_rewards.items():
    # 如果数据长度小于窗口长度，建议直接绘制原始数据或调整窗口大小
    if len(rewards) >= rolling_length:
        rewards_ma = moving_average(rewards, rolling_length, mode="valid")
        axs[0, 0].plot(range(len(rewards_ma)), rewards_ma, label=key)
    else:
        axs[0, 0].plot(range(len(rewards)), rewards, label=key)
axs[0, 0].set_title("Episode Rewards (Rolling Average)")
axs[0, 0].set_xlabel("Episode")
axs[0, 0].set_ylabel("Reward")
axs[0, 0].legend(fontsize='small', loc='upper right')

# 绘制剧集长度的滚动平均曲线
for key, lengths in all_lengths.items():
    if len(lengths) >= rolling_length:
        lengths_ma = moving_average(lengths, rolling_length, mode="valid")
        axs[0, 1].plot(range(len(lengths_ma)), lengths_ma, label=key)
    else:
        axs[0, 1].plot(range(len(lengths)), lengths, label=key)
axs[0, 1].set_title("Episode Lengths (Rolling Average)")
axs[0, 1].set_xlabel("Episode")
axs[0, 1].set_ylabel("Length")
axs[0, 1].legend(fontsize='small', loc='upper right')

# 绘制平均 Loss 的滚动平均曲线
for key, losses in all_losses.items():
    axs[1, 0].plot(range(len(losses)), losses, label=key)
axs[1, 0].set_title("Episode Average Loss (Rolling Average)")
axs[1, 0].set_xlabel("Episode")
axs[1, 0].set_ylabel("Loss")
axs[1, 0].legend(fontsize='small', loc='upper right')

# 绘制平均 Q 值的滚动平均曲线
for key, qs in all_q_values.items():
    axs[1, 1].plot(range(len(qs)), qs, label=key)
axs[1, 1].set_title("Episode Average Q Value (Rolling Average)")
axs[1, 1].set_xlabel("Episode")
axs[1, 1].set_ylabel("Q Value")
axs[1, 1].legend(fontsize='small', loc='upper right')


now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
plt.tight_layout()
plt.savefig(f"""{formatted_time}_comparison_all_metrics.jpg""")
plt.show()


# Hyperparameter Combinations Save
summary_file = f"""{formatted_time}_config.csv"""
with open(summary_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["learning_rate", "gamma", "epsilon_decay", "dt", "D", "model", "truncation", "guided"])
    for res in results_summary:
        writer.writerow([res["learning_rate"], res["gamma"], res["epsilon_decay"],
                         res["dt"], res["D"], res["model"], res["truncation"], res["guided"]])

