import gym
import numpy as np
import random
import math
import cv2
import imageio
from gym import spaces
from matplotlib import pyplot as plt

# Each state is an image. State space is 2D.

PHI_MAX = 30

X_STATES = 400
TILT_STATES = 101
NUM_OF_ACTIONS = 5
IN_DIM = (1, 84, 84)
MASK_FILE = '/content/drive/My Drive/UBC Research/mask.png'
INTERPOLATION = cv2.INTER_NEAREST

class navigate2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, path, is_test=0):
        self.state_array = np.load(path)
        self.state_array = self.state_array.transpose(0, 2, 1)
        self.is_test = is_test
        self.nbEpisode = 1
        self.action_space = spaces.Discrete(NUM_OF_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=IN_DIM, dtype='uint8')
        self.flag = False
        self.done = False
        self.state = None
        self.x_index = random.randint(0, X_STATES - 1)
        self.tilt_index = random.randint(0, TILT_STATES - 1)

        self.mask = imageio.imread(MASK_FILE)[:,:,1]
        self.mask = np.uint8(self.mask/255)
        self.mask = cv2.resize(self.mask, dsize=(400, 400), interpolation=INTERPOLATION)

        dims = self.state_array.shape
        self.x0 = dims[0]
        self.y0 = dims[1]
        self.z0 = dims[2]

    def step(self, action):
        print(action)
        state, reward = self.take_action(action)
        return state, reward, self.done, {}

    def reset(self):
        print('Episode: ' + str(self.nbEpisode))
        self.flag = False
        self.done = False

        if self.nbEpisode < 10 and self.is_test == 0:
            self.x_index = random.randint(190, 210)
            self.tilt_index = random.randint(42, 58)
        elif self.nbEpisode < 30 and self.is_test == 0:
            self.x_index = random.randint(185, 215)
            self.tilt_index = random.randint(40,60)
        elif self.nbEpisode < 60 and self.is_test == 0:
            self.x_index = random.randint(170, 230)
            self.tilt_index = random.randint(35, 65)
        elif self.nbEpisode < 100 and self.is_test == 0:
            self.x_index = random.randint(140, 260)
            self.tilt_index = random.randint(30, 70)
        elif self.nbEpisode < 140 and self.is_test == 0:
            self.x_index = random.randint(100, 300)
            self.tilt_index = random.randint(20, 80)
        elif self.nbEpisode < 200 and self.is_test == 0:
            self.x_index = random.randint(50, 350)
            self.tilt_index = random.randint(10, 90)
        else:
            self.x_index = random.randint(0, X_STATES - 1)
            self.tilt_index = random.randint(0, TILT_STATES - 1)

        self.state = self.get_slice(0, self.tilt_index*2/100 - 1, self.x_index*2/399 - 1)
        state = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)
        state = state[np.newaxis, :, :]

        return state

    def render(self, mode='human'):
        plt.imshow(self.state, cmap='gray', vmin=0, vmax=255)
        plt.show()
        print('Position: ( ' + str(self.x_index) + ', ' + str(self.tilt_index) + ')')
        cv2.waitKey(0)

    def take_action(self, action):
        if action == 0:  # Wait
            temp_x = self.x_index
            temp_tilt = self.tilt_index
        elif action == 1:  # Left
            temp_x = self.x_index - 10
            temp_tilt = self.tilt_index
        elif action == 2:  # Right
            temp_x = self.x_index + 10
            temp_tilt = self.tilt_index
        elif action == 3:  # Up
            temp_x = self.x_index
            temp_tilt = self.tilt_index + 5
        else:  # Down
            temp_x = self.x_index
            temp_tilt = self.tilt_index - 5

        if temp_x < 0 or temp_x > X_STATES - 1 or temp_tilt < 0 or temp_tilt > TILT_STATES - 1:
            obs = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)[np.newaxis, :, :]
            reward = -0.1
        else:
            reinf = np.abs(self.x_index - 199) + np.abs(self.tilt_index - 50) > np.abs(temp_x - 199) + np.abs(temp_tilt - 50)
            self.x_index = temp_x
            self.tilt_index = temp_tilt
            self.flag = 194 < self.x_index < 205 and 47 < self.tilt_index < 53
            self.done = self.flag and action == 0
            self.state = self.get_slice(0, self.tilt_index*2/100 - 1, self.x_index*2/399 - 1)
            obs = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)[np.newaxis, :, :]
            reward = 0.1*(1 - self.done)*(-1 + 2*reinf) + self.done/5

        self.nbEpisode = self.nbEpisode + 1*self.done

        return obs, reward

    def get_bounding_box(self, theta, phi, dist):
        h1 = [self.x0 / 2 + self.y0 / 2 * math.sin(theta) + dist,  # *math.cos(theta),
              self.y0 / 2 - self.y0 / 2 * math.cos(theta)]  # + dist*math.sin(theta)]

        h2 = [self.x0 / 2 - self.y0 / 2 * math.sin(theta) + dist,  # *math.cos(theta),
              self.y0 / 2 + self.y0 / 2 * math.cos(theta)]  # + dist*math.sin(theta)]

        z_min = 0  # self.z0 / 2 - self.z0 / 2 * math.cos(phi)
        z_max = self.z0 * math.cos(phi)  # self.z0 / 2 + self.z0 / 2 * math.cos(phi)
        return h1, h2, z_min, z_max

    def get_slice(self, theta_n, phi_n, dist_n):
        theta = theta_n * math.pi
        phi = math.radians(phi_n * PHI_MAX)
        dist = dist_n * self.x0 / 2  # +/- 104 pixels

        # --- 1: Get bounding box dims ---
        h1, h2, z_min, z_max = self.get_bounding_box(theta=theta, phi=phi, dist=dist)
        w = self.y0
        h = self.z0

        # --- 2: Extract slice from volume ---
        # Get x_i and y_i for current layer
        x_offsets = np.linspace(0, h, h) * math.sin(phi) * math.cos(
            theta)  # np.linspace(-h/2, h/2, h) * math.sin(phi) * math.cos(theta)
        y_offsets = np.linspace(0, h, h) * math.sin(phi) * math.sin(
            theta)  # np.linspace(-h/2, h/2, h) * math.sin(phi) * math.sin(theta)

        # Tile and transpose
        x_offsets = np.transpose(np.tile(x_offsets, (w, 1)))
        y_offsets = np.transpose(np.tile(y_offsets, (w, 1)))

        x_i = np.tile(np.linspace(h1[0], h2[0], w), (h, 1))
        y_i = np.tile(np.linspace(h1[1], h2[1], w), (h, 1))

        x_i = np.array(np.rint(x_i + x_offsets), dtype='int')
        y_i = np.array(np.rint(y_i + y_offsets), dtype='int')

        # Don't forget to include the index offset from z!
        z_i = np.transpose(np.tile(np.linspace(z_min, z_max, h), (w, 1)))
        z_i = np.array(np.rint(z_i), dtype='int')

        # Flatten
        flat_inds = np.ravel_multi_index((x_i, y_i, z_i), (self.x0, self.y0, self.z0), mode='clip')

        # Fill in entire slice at once
        the_slice = np.take(self.state_array, flat_inds)

        # --- 3: Mask slice ---
        the_slice = np.multiply(the_slice, self.mask)
        return the_slice
