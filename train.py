import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from cartpole_env import CartPoleEnv
from dqn import DQN


def train(env, net, optimizer, loss_fn, episodes=1000):
    gamma = 0.99
    epsilon = 0.1
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = net(state_tensor)
            if random.random() > epsilon:
                action = torch.argmax(q_values).item()
            else:
                action = random.choice([0, 1])

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            next_q_values = net(next_state_tensor)
            max_next_q_value = torch.max(next_q_values).item()

            target_q_value = reward + gamma * max_next_q_value * (1 - done)
            target_q_values = q_values.clone()
            target_q_values[0, action] = target_q_value

            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        print(f"Episode {episode}, Total Reward: {total_reward}")
        if episode % 100 == 0:
            torch.save(net.state_dict(), "dqn_model.pth")

env = CartPoleEnv()
net = DQN()
optimizer = optim.Adam(net.parameters())
loss_fn = nn.MSELoss()

train(env, net, optimizer, loss_fn)
