import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from DuelingDQN import DuelingNetwork
from collections import deque


class DeepQAgent:
    def __init__(
        self,
        env,
        state_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        buffer_size,
        batch_size,
        tau,
        model
    ):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.batch_size = batch_size
        self.tau = tau
        self.model = model

        self.memory = deque(maxlen=buffer_size)

        # Define networks
        self.device = torch.device("cuda")
        self.policy_net = self.build_network(state_dim, action_dim).to(self.device)
        self.target_net = self.build_network(state_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

        self.steps_done = 0

        random.seed(888)
        np.random.seed(888)
        torch.manual_seed(888)
        torch.cuda.manual_seed(888)


    def build_network(self, input_dim, output_dim):
        if self.model == "DQN":
            return nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU6(),
                nn.Linear(128, 128),
                nn.ReLU6(),
                nn.Linear(128, output_dim)
            )
        elif self.model == "DuelingDQN":
            return DuelingNetwork(input_dim, output_dim)


    @torch.no_grad()
    def get_action(self, state):
        self.steps_done += 1
        self.epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_end)
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return int(torch.argmax(self.policy_net(state)))


    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                        1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


    def update(self):
        if len(self.memory) < self.batch_size:
            return (None, None)

        # Sample a batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q values
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = self.criterion(current_q, target_q)
        loss_val = loss.item()
        avg_q = current_q.mean().item()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.update_target_network()

        return loss_val, avg_q
