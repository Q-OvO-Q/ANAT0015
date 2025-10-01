import gymnasium as gym
from config import config
import chemostat_env
from scipy.integrate import odeint
import itertools
import copy
import csv
import os


def simulate_one_step(x, S, action, dt, config):
    D_val = config["D"]
    if action == 1:
        x = x - D_val * x
        S = S - D_val * S + D_val * config["S_in"]

    def chemostat_model(state, t):
        x_val, S_val = state
        mu = config["mu_max"] * S_val / (config["Ks"] + S_val)
        dxdt = mu * x_val
        dSdt = - (mu / config["gamma"]) * x_val
        return [dxdt, dSdt]

    t_span = [0, dt]
    result = odeint(chemostat_model, [x, S], t_span)
    x_new, S_new = result[-1]
    return x_new, S_new


def mpc_choose_action(x, S, target_x, dt, config):
    lower_bound = config["lower_bound"]
    upper_bound = config["upper_bound"]
    penalty = 1e6  # 超出上下界时的惩罚

    # Action 0 No Dilution
    x0, S0 = simulate_one_step(x, S, 0, dt, config)
    cost0 = abs(x0 - target_x)
    if x0 < lower_bound or x0 > upper_bound:
        cost0 += penalty

    # Action 1 Dilution
    x1, S1 = simulate_one_step(x, S, 1, dt, config)
    cost1 = abs(x1 - target_x)
    if x1 < lower_bound or x1 > upper_bound:
        cost1 += penalty

    # Choose low cost
    if cost0 <= cost1:
        return 0, x0, S0, cost0, cost1
    else:
        return 1, x1, S1, cost0, cost1


def run_mpc_episode(env, dt, config, render=False):
    state, info = env.reset(seed=888)
    total_reward = 0
    done = False
    scale_factor = config["scale_factor"]
    current_info = info["info"]
    x = current_info[0]
    S = current_info[2]
    target_x = state[3] * scale_factor

    log_file = f"MPC.csv"
    log_exists = os.path.exists(log_file)
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not log_exists:
            writer.writerow(["Action", "Reward", "Done", "State", "Info"])

    log_data = []
    step = 0
    while not done:
        action, x_pred, S_pred, cost0, cost1 = mpc_choose_action(x, S, target_x, dt, config)
        log_data.append(
            f"Step {step}: x={x:.2e}, S={S:.2e}, target_x={target_x:.2e}, chosen action={action}, "
            f"predicted x={x_pred:.2e}, cost0={cost0:.2e}, cost1={cost1:.2e}"
        )
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Update
        current_info = info["info"]
        x = current_info[0]
        S = current_info[2]
        target_x = state[3] * scale_factor
        step += 1

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([action, reward, done, state.tolist(), info])

        if render:
            env.render()

    print("\n".join(log_data))
    print(f"Episode finished. Total Reward: {total_reward}\n")

    return total_reward


def main():
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
        env = gym.make("ChemostatEnv-v0", config=local_config)
        dt_val = local_config["dt"]

        n_episodes = 1
        rewards = []

        for ep in range(n_episodes):
            ep_reward = run_mpc_episode(env, dt_val, local_config)
            rewards.append(ep_reward)
            print(f"Episode {ep}, Reward: {ep_reward}")

    env.close()


if __name__ == "__main__":
    main()
