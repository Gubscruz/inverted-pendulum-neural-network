import numpy as np
import pygame
import math



class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.state = None


    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state


    def step(self, action):
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        done = x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        done = bool(done)

        reward = 1.0 if not done else 0.0

        return np.array(self.state), reward, done, {}


    def render(self, screen):
        screen.fill((255, 255, 255))

        cart_y = 300
        cart_w = 50
        cart_h = 30

        pole_len = 100
        pole_w = 10

        x = self.state[0] * 100 + 400
        theta = self.state[2]

        cart_rect = pygame.Rect(x - cart_w // 2, cart_y - cart_h // 2, cart_w, cart_h)
        pygame.draw.rect(screen, (0, 0, 0), cart_rect)

        pole_x = x + pole_len * math.sin(theta)
        pole_y = cart_y - pole_len * math.cos(theta)
        pygame.draw.line(screen, (255, 0, 0), (x, cart_y), (pole_x, pole_y), pole_w)

        pygame.display.flip()
