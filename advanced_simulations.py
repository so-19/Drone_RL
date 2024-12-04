import os
import pybullet as p
import numpy as np
import time
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1. / 240.)
        p.setRealTimeSimulation(0)

        self.drone = p.loadURDF("drone.urdf", basePosition=[0, 0, 2])

        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([20, 20, 20, 20]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        self.max_steps = 1000
        self.current_step = 0

    def step(self, action):
        self.current_step += 1

        motor_positions = [
            [0.2, 0.2, 0], [-0.2, 0.2, 0],
            [0.2, -0.2, 0], [-0.2, -0.2, 0]
        ]

        for i in range(4):
            force = [0, 0, float(action[i])]
            p.applyExternalForce(self.drone, i, force, motor_positions[i], p.LINK_FRAME)
            p.resetJointState(self.drone, i, action[i], 0)

        for _ in range(2):
            p.stepSimulation()

        state = self._get_state()

        reward = self._calculate_reward(state)

        done = self._is_done(state)

        return state, reward, done, {}

    def _get_state(self):
        pos, ori = p.getBasePositionAndOrientation(self.drone)
        vel, angvel = p.getBaseVelocity(self.drone)
        euler = p.getEulerFromQuaternion(ori)

        normalized_step = self.current_step / self.max_steps

        return np.array(list(pos) + list(euler) + list(vel) + list(angvel) + [normalized_step])

    def _calculate_reward(self, state):
        pos = state[:3]
        euler = state[3:6]
        vel = state[6:9]
        angvel = state[9:12]

        height_target = 2.0
        height_error = abs(pos[2] - height_target)
        orientation_error = np.sum(np.abs(euler))

        vel_penalty = np.sum(np.abs(vel))
        angvel_penalty = np.sum(np.abs(angvel))

        # Combine rewards
        reward = (
            10.0 * np.exp(-orientation_error) +  # Orientation reward
            5.0 * np.exp(-height_error) -  # Height reward
            0.1 * vel_penalty -  # Velocity penalty
            0.1 * angvel_penalty  # Angular velocity penalty
        )

        # Bonus for stable hovering
        if (height_error < 0.1 and
            orientation_error < 0.1 and
            vel_penalty < 0.1 and
            angvel_penalty < 0.1):
            reward += 50.0

        return reward

    def _is_done(self, state):
        pos = state[:3]
        euler = state[3:6]

        if pos[2] < 0.1 or pos[2] > 4.0:  # Height limits
            return True
        if abs(pos[0]) > 3.0 or abs(pos[1]) > 3.0:  # Position limits
            return True
        if any(abs(angle) > math.pi for angle in euler):  # Extreme angles
            return True
        if self.current_step >= self.max_steps:  # Max steps reached
            return True

        return False

    def reset(self):
        self.current_step = 0
        start_pos = [0, 0, 2]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.drone, start_pos, start_ori)
        p.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])

        return self._get_state()


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)


class DroneController:
    def __init__(self):
        self.env = DroneEnv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = 13
        self.action_dim = 4
        self.num_discrete_actions = 5

        self.policy_net = DQN(self.state_dim, self.action_dim * self.num_discrete_actions).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim * self.num_discrete_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = deque(maxlen=100000)

        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 10000
        self.steps_done = 0

        self.episode_rewards = []
        self.avg_rewards = []
        self.best_reward = float('-inf')

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10))

    def _discretize_action(self, action_indices):
        discrete_values = torch.linspace(0, 20, self.num_discrete_actions)
        actions = []

        for i in range(self.action_dim):
            idx = action_indices[i]
            actions.append(discrete_values[idx])

        return np.array(actions)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                output = self.policy_net(state_tensor).view(self.action_dim, -1)
                action_indices = output.max(1)[1]
                return self._discretize_action(action_indices)
        else:
            return np.random.uniform(0, 20, size=self.action_dim)

    def update_plots(self):
        self.ax1.clear()
        self.ax1.plot(self.episode_rewards)
        self.ax1.set_title('Episode Rewards')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')

        if len(self.episode_rewards) > 0:
            self.avg_rewards.append(np.mean(self.episode_rewards[-100:]))
            self.ax2.clear()
            self.ax2.plot(self.avg_rewards)
            self.ax2.set_title('100-Episode Moving Average Reward')
            self.ax2.set_xlabel('Episode')
            self.ax2.set_ylabel('Average Reward')

        plt.pause(0.001)

    def train(self, num_episodes=1000):
        print(f"Training on device: {self.device}")

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0

            for t in range(self.env.max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                if len(self.memory) >= self.batch_size:
                    self.optimize_model()

                if done:
                    break

            self.episode_rewards.append(total_reward)

            if episode % 10 == 0:
                self.update_plots()
                print(f"Episode {episode}/{num_episodes}, Reward: {total_reward:.2f}, "
                      f"Avg Reward (100 ep): {np.mean(self.episode_rewards[-100:]):.2f}")

            if episode % 20 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'reward': total_reward,
                }, f'best_drone_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
                print(f"New best reward: {total_reward:.2f}")

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.FloatTensor(np.array(batch[1])).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch[2])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch[4])).to(self.device)

        current_q_values = self.policy_net(state_batch)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            next_q_values = next_q_values.view(self.batch_size, self.action_dim, -1).max(2)[0]

            expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values.max(1)[0]

        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values.view(self.batch_size, -1), expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=100)
        self.optimizer.step()


def main():
    controller = DroneController()
    try:
        controller.train(num_episodes=1000)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        p.disconnect()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
