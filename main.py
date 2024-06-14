import pygame
import torch
from cartpole_env import CartPoleEnv
from dqn import DQN

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

env = CartPoleEnv()
net = DQN()
net.load_state_dict(torch.load("dqn_model.pth"))

env.reset()
done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    state_tensor = torch.FloatTensor(env.state).unsqueeze(0)
    q_values = net(state_tensor)
    action = torch.argmax(q_values).item()

    env.step(action)
    env.render(screen)
    clock.tick(30)

pygame.quit()
