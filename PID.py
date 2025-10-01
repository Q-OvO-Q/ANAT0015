import gymnasium as gym
from config import config
import chemostat_env
import itertools
import copy
import csv
import os


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = None

    def update(self, measurement, dt):
        error = measurement - self.setpoint
        self.integral += error * dt
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self):
        self.integral = 0.0
        self.prev_error = None


def run_pid_episode(env, pid, dt, render=False):
    state, info = env.reset(seed=888)
    total_reward = 0
    done = False
    scale_factor = config['scale_factor']
    log_data = []
    step = 0

    log_file = f"PID.csv"
    log_exists = os.path.exists(log_file)
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not log_exists:
            writer.writerow(["Action", "Reward", "Done", "State", "Info"])

    while not done:
        current_info = info["info"]
        x = current_info[0]
        target_x = state[3] * scale_factor

        pid.setpoint = target_x

        control_output = pid.update(x, dt)
        action = 1 if control_output > 0 else 0

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([action, reward, done, state.tolist(), info])

        log_data.append(
            f"Step {step}: x={x:.2e}, target_x={target_x:.2e}, control_output={control_output:.2e}, action={action}, reward={reward}"
        )
        step += 1

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

        # 由于x和target_x的量级大约在1e11左右，因此这里选择一个较小的Kp
        pid = PIDController(Kp=1e-10, Ki=0, Kd=0)

        n_episodes = 1
        rewards = []

        for ep in range(n_episodes):
            pid.reset()
            ep_reward = run_pid_episode(env, pid, dt_val, render=False)
            rewards.append(ep_reward)
            print(f"Episode {ep}, Reward: {ep_reward}")

    env.close()


if __name__ == "__main__":
    main()
