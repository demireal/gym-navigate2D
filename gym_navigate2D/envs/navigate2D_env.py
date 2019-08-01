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

    def __init__(self, path, is_test=0, is_same=0):
        self.state_array = np.load(path)
        self.state_array = self.state_array.transpose(0, 2, 1)
        self.data = self.state_array
        self.is_test = is_test
        self.is_same = is_same
        self.nbEpisode = 1
        self.action_space = spaces.Discrete(NUM_OF_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=IN_DIM, dtype='uint8')
        self.flag = False
        self.done = False
        self.state = None
        self.x_index = random.randint(0, X_STATES - 1)
        self.tilt_index = random.randint(0, TILT_STATES - 1)

        self.mask = imageio.imread(MASK_FILE)[:, :, 1]
        self.mask = np.uint8(self.mask/255)
        self.mask = cv2.resize(self.mask, dsize=(400, 400), interpolation=INTERPOLATION)

        dims = self.state_array.shape
        self.x0 = dims[0]
        self.y0 = dims[1]
        self.z0 = dims[2]

    def step(self, action):
        state, reward = self.take_action(action)
        return state, reward, self.done, {}

    def reset(self):
        self.flag = False
        self.done = False
        self.state_array = self.data + random.randint(0, 60)*np.random.randn(400, 400, 400)
        self.state_array = np.clip(self.state_array, 0, 255)
        self.state_array = np.array(self.state_array, np.uint8)

        if self.nbEpisode < 10 and self.is_test == 0 and self.is_same == 0:
            self.x_index = random.randint(180, 220)
            self.tilt_index = random.randint(40, 60)
        elif self.nbEpisode < 20 and self.is_test == 0 and self.is_same == 0:
            self.x_index = random.randint(140, 260)
            self.tilt_index = random.randint(30, 70)
        elif self.nbEpisode < 30 and self.is_test == 0 and self.is_same == 0:
            self.x_index = random.randint(100, 300)
            self.tilt_index = random.randint(10, 90)
        else:
            self.x_index = random.randint(0, X_STATES - 1)
            self.tilt_index = random.randint(0, TILT_STATES - 1)

        self.state = self.get_slice(0, self.tilt_index * 2 / 100 - 1, self.x_index * 2 / 399 - 1)
        state = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)
        state = state[np.newaxis, :, :]

        return state

    def render(self, mode='human'):
        plt.imshow(self.state, cmap='gray', vmin=0, vmax=255)
        plt.show()
        cv2.waitKey(0)
        print('(TILT, X): ( ' + str(self.tilt_index) + ', ' + str(self.x_index) + ')')

    def take_action(self, action):

        temp_x = int((self.x_index - 8) * (action == 1) + (self.x_index + 8) * (action == 2) + self.x_index * (action != 1 and action != 2))
        temp_tilt = int((self.tilt_index - 3) * (action == 3) + (self.tilt_index + 3) * (action == 4) + self.tilt_index * (action != 3 and action != 4))

        if temp_x < 0 or temp_x > (X_STATES - 1) or temp_tilt < 0 or temp_tilt > (TILT_STATES - 1):
            obs = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)[np.newaxis, :, :]
            reward = -0.1
        else:
            reinf = np.abs(self.x_index - 199) + np.abs(self.tilt_index - 50) > np.abs(temp_x - 199) + np.abs(temp_tilt - 50)
            self.x_index = temp_x
            self.tilt_index = temp_tilt
            self.flag = 195 < self.x_index < 204 and 48 < self.tilt_index < 52
            self.done = self.flag and action == 0
            self.state = self.get_slice(0, self.tilt_index*2/100 - 1, self.x_index*2/399 - 1)
            obs = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)[np.newaxis, :, :]
            reward = 0.1*(1 - self.done)*(-1 + 2*reinf) + self.done

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
