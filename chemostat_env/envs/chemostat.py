import gymnasium as gym
import random
from gymnasium import spaces
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Optional
import datetime
import json
import pygame


class ChemostatEnv(gym.Env):

    def __init__(self, config, render=False):
        super(ChemostatEnv, self).__init__()
        self.config = config
        self.dt = self.config["dt"]
        self.D = self.config['D']
        self.upper_bound = self.config['upper_bound']
        self.lower_bound = self.config['lower_bound']
        self.max_steps = int(100 / self.dt)
        self.mode = config['mode']
        self.truncation = config['truncation']
        self.guided = config['guided']

        # Define action and observation spaces
        # Action: Discrete actions (0 = no dilution, 1 = dilute)
        self.action_space = spaces.Discrete(2)

        # Observations: state of the environment (here we can observe the number of cells and its trajectory)
        # val - [scaled_X, scaled_dXdt, target_x, target_y]
        # min - [0, -dilution_rate*carry_capacity/scale_factor, -1, 0]
        # max - [carry_capacity/scale_factor, max_growth_rate*carry_capacity/scale_factor, 50, carry_capacity/scale_factor]

        if self.mode == 'survive':
            self.observation_space = spaces.Box(low=np.array([0, -0.1]),
                                                high=np.array([1, 0.5]),
                                                dtype=np.float64)
        else:
            self.observation_space = spaces.Box(low=np.array([0, -0.1, -1, 0]),
                                                high=np.array([1, 0.5, 50, 1]),
                                                dtype=np.float64)

        # self._initial_state() #Used to plot the model with the given parameters

        self.window_size = 256
        self.fps = 60
        self.render = render
        self.window = None
        self.clock = None

        self.reset(seed=888)


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, **kwargs):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        super().reset(seed=seed)
        # Reset simulation state
        self.x = self.config['x']  # Initial microbial density
        self.S = 0.1               # Initial substrate concentration ############!!!!!!!!############
        model_return = self._chemostat_model([self.x, self.S], 0)
        self.acc = model_return[0]

        # Use truncation otherwise sometimes it won't stop at all
        self.current_step = 0
        self.score = 0

        # Generate initial target
        self.pipe_gap = self.config['pipe_gap']
        self.pipe_t, self.target_x = self.get_random_pipe()

        observation = self._get_obs()
        info = self._get_info()

        if self.render:
            self._render_frame()

        # Return initial observation
        return observation, info


    def step(self, action):
        # Apply agent's action
        if action == 1:
            self.x = self.x - self.D * self.x
            self.S = self.S - self.D * self.S + self.D * self.config['S_in']

        # Update microbial state using chemostat_model
        results = odeint(self._chemostat_model, [self.x, self.S], [0, self.dt])
        self.x = results[1, 0]
        self.S = results[1, 1]
        model_return = self._chemostat_model([self.x, self.S], 0)
        self.acc = model_return[0]

        # Reward
        reward = 0

        if self.mode == 'survive':
            terminated = self.x < self.lower_bound or self.x > self.upper_bound
            if terminated:
                reward = -1000
            else:
                reward = 1
        elif self.mode == 'moving_target':
            terminated = self.x < self.lower_bound or self.x > self.upper_bound
            if terminated:
                reward = -1000
            else:
                scaled_distance = abs(self.x - self.target_x) / self.target_x
                reward = (1 - scaled_distance) / 10
        elif self.mode == 'flappy_bird':
            terminated_1 = self.x < self.lower_bound or self.x > self.upper_bound
            terminated_2 = self.pipe_t <= 0 and (self.x < (self.target_x - self.pipe_gap) or self.x > (self.target_x + self.pipe_gap))
            terminated = terminated_1 or terminated_2
            if terminated:
                reward = -5
            elif self.pipe_t <= 0:
                reward = 1
            else:
                if self.guided:
                    scaled_distance = abs(self.x - self.target_x) / self.target_x
                    reward = (1 - scaled_distance) / 10
                else:
                    reward = 0

        if self.truncation:
            truncated = self.current_step >= self.max_steps
        else:
            truncated = False
        self.current_step += 1
        self.score = self.score + reward

        observation = self._get_obs()
        info = self._get_info()

        # Evolve the target
        # if a target exists, move it one time_step closer
        if self.pipe_t > 0:  # target_x
            self.pipe_t = self.pipe_t - self.dt
        # else if a target does not exist, create a new target at a random height and distance from the current timepoint
        else:
            self.pipe_t, self.target_x = self.get_random_pipe()

        if self.render:
            self._render_frame()

        return observation, reward, terminated, truncated, info


    def _chemostat_model(self, y, t):
        x, S = y
        mu = self.config['mu_max'] * S / (self.config['Ks'] + S)  # Specific growth rate
        dx_dt = mu * x
        dS_dt = - (mu / self.config['gamma']) * x
        return [dx_dt, dS_dt]


    def get_random_pipe(self):
        pipe_t = random.randint(1, 1)  ############################
        target_x = (3 * random.random() + 1.5) * 1e11  ####可改动参数####
        return pipe_t, target_x


    def _get_info(self):
        if self.mode == 'survive':
            return {'info': np.array([self.x, self.acc, self.S], dtype='float64')}
        else:
            return {'info': np.array([self.x, self.acc, self.S, self.x - self.target_x], dtype='float64')}


    def _get_obs(self):
        scale_factor = self.config['scale_factor']
        normalized_x = self.x / scale_factor
        normalized_acc = self.acc / scale_factor
        if self.mode == 'survive':
            return np.array([normalized_x, normalized_acc], dtype='float64')
        else:
            normalized_target_x = self.target_x / scale_factor
            return np.array([normalized_x, normalized_acc, self.pipe_t, normalized_target_x], dtype='float64')


    def render(self):
        return self._render_frame()

  
    def _render_frame(self):
        if self.window is None and self.render:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # fill white

        # First we draw the target defined by target_x and target_y
        if self.mode == "flappy_bird":
            lower_bound, upper_bound = (self.target_x - self.pipe_gap), (self.target_x + self.pipe_gap)
            height = upper_bound - lower_bound
            # draw walls
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                (
                    (self.pipe_t * 10) + 10,  # x (left) position
                    upper_bound / self.config['scale_factor'] * self.window_size,  # y (top) position
                    5,  # width
                    self.window_size - upper_bound / self.config['scale_factor']  # height
                ),
            )
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                (
                    (self.pipe_t * 10) + 10,  # x (left) position
                    0,  # y (top) position
                    5,  # width
                    lower_bound / self.config['scale_factor'] * self.window_size  # height
                ),
            )
        elif self.mode == "moving_target":
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                (
                    10,  # x (left) position
                    self.target_x / self.config['scale_factor'] * self.window_size,  # y (top) position
                    (self.pipe_t * 10) + 10,  # width
                    1  # height
                ),
            )

        # Draw the agent with y position at state.X
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            center=(
                10,  # x position
                self.x / self.config['scale_factor'] * self.window_size  # y position
            ),
            radius=5
        )

        # draw upper and lower bounds
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            (
                0,  # x (left) position
                self.upper_bound / self.config['scale_factor'] * self.window_size,  # y (top) position
                self.window_size,  # width
                1  # height
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            (
                0,  # x (left) position
                self.lower_bound / self.config['scale_factor'] * self.window_size,  # y (top) position
                self.window_size,  # width
                1  # height
            ),
        )

        # Print the state of the environment
        font = pygame.font.Font(None, 12)
        if self.mode == "survive":
            text = font.render(
                f"X: {self.x:.2e}, dX/dt: {self.acc:.2e}, Score: {self.score:.2e}",
                True,
                (0, 0, 0))
        else:
            text = font.render(
                f"X: {self.x:.2e}, S: {self.S:.2e}, "
                f"Target: ({self.pipe_t:.2e}, {self.target_x:.2e}, )"
                f"Score: {self.score:.2e}",
                True,
                (0, 0, 0))
        canvas.blit(text, (10, 230))

        if self.render:
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.fps)

  
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


    def _initial_state(self):
        t = np.arange(0, 5, 0.01)
        results = odeint(self._chemostat_model, [self.config['x'], self.config['S_in']], t)
        x_result = results[:, 0]
        S_result = results[:, 1]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Substrate Concentration (S)
        color1 = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Substrate Concentration (S)', color=color1)
        ax1.plot(t, S_result, color=color1, label="Substrate Concentration (S)")
        ax1.tick_params(axis='y', labelcolor=color1)

        # Plot Biomass Concentration (x) on the same graph
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Biomass Concentration (x)', color=color2)
        ax2.plot(t, x_result, color=color2, label="Biomass Concentration (x)")
        ax2.tick_params(axis='y', labelcolor=color2)

        fig.tight_layout()
        plt.grid(True)
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"""{formatted_time}.png""")
        with open(f"""{formatted_time}_config_1.txt""", "w", encoding="utf-8") as f:
            json_str = json.dumps(self.config, ensure_ascii=False, indent=4)
            f.write(json_str)
