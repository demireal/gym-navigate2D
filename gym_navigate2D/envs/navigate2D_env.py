import gym
import imageio
import glob
import numpy as np
import random
import pandas as pd


from time import sleep
from IPython.display import clear_output
from gym import error, spaces, utils
from gym.utils import seeding
from PIL import Image
from matplotlib import pyplot as plt

# Each state is an image. State space is 2D.

INPUT_SHAPE = (50, 50, 1)
X_STATES = 20
Y_STATES = 113
NUM_OF_ACTIONS = 5
STATE_ARRAY = np.zeros((INPUT_SHAPE[0], INPUT_SHAPE[1], X_STATES, Y_STATES))

DF = pd.read_csv('/content/drive/My Drive/xy.csv')
DISTANCES = DF.values
for index_1, str_arr in enumerate(DISTANCES):
  for index_2, strr in enumerate(str_arr):
     DISTANCES [index_1, index_2] = float(strr.replace(',','.'))

      
class navigate2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, path, is_test=0):
        STATE_ARRAY = np.load(path)
        self.is_test = is_test
        self.x_index = random.randint(0, X_STATES - 1)
        self.y_index = random.randint(0, Y_STATES - 1)
        self.state = STATE_ARRAY[:, :, self.x_index, self.y_index, np.newaxis]
        self.flag = False
        self.done = False
        self.nbEpisode = 1
        self.action_space = spaces.Discrete(NUM_OF_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1, shape=INPUT_SHAPE, dtype='float32')

    def step(self, action):
        self.state, reward = self.take_action(action)
        return self.state, reward, self.done, {}

    def reset(self):
        print('Episode: ' + str(self.nbEpisode))
        if self.nbEpisode < 10 and self.is_test == 0:
          self.x_index = random.randint(5, 9)
          self.y_index = random.randint(50, 66)
        elif self.nbEpisode < 25 and self.is_test == 0:
          self.x_index = random.randint(4, 11)
          self.y_index = random.randint(40, 76)
        elif self.nbEpisode < 45 and self.is_test == 0:
          self.x_index = random.randint(3, 13)
          self.y_index = random.randint(30, 86)
        elif self.nbEpisode < 70 and self.is_test == 0:
          self.x_index = random.randint(2, 15)
          self.y_index = random.randint(20, 96)
        elif self.nbEpisode < 100 and self.is_test == 0:
          self.x_index = random.randint(1, 17)
          self.y_index = random.randint(10, 106)
        else:
          self.x_index = random.randint(0, X_STATES - 1)
          self.y_index = random.randint(0, Y_STATES - 1)
        self.state = STATE_ARRAY[:, :, self.x_index, self.y_index, np.newaxis]
        self.flag = False
        self.done = False
        return self.state

    def render(self, mode='human'):
        plt.imshow(255*self.state[:, :, 0], cmap='gray', vmin=0, vmax=255)
        plt.show()
        print('Position: ( ' + str(self.x_index) + ', ' + str(self.y_index) + ')')
        sleep(1)
        clear_output()

    def take_action(self, action):
        if action == 0:  # Wait
            tmp_x_index = self.x_index
            tmp_y_index = self.y_index
        elif action == 1:  # Left
            tmp_x_index = self.x_index + 1
            tmp_y_index = np.argmin(np.abs(DISTANCES[self.x_index, self.y_index] - DISTANCES[np.mod(tmp_x_index, X_STATES), :]))
        elif action == 2:  # Right
            tmp_x_index = self.x_index - 1
            tmp_y_index = np.argmin(np.abs(DISTANCES[self.x_index, self.y_index] - DISTANCES[np.mod(tmp_x_index, X_STATES), :]))
        elif action == 3:  # Up
            tmp_x_index = self.x_index
            tmp_y_index = self.y_index + 3
        else:  # Down
            tmp_x_index = self.x_index
            tmp_y_index = self.y_index - 3
              
        if tmp_x_index < 0 or tmp_x_index > X_STATES - 1 or tmp_y_index < 0 or tmp_y_index > Y_STATES - 1:
            obs = self.state
            reward = -0.1
        else:
            right_choice = (np.square(DISTANCES[self.x_index, self.y_index] - DISTANCES[7, 58]) + np.square((self.x_index - 10)*2 + 4)) > (np.square(DISTANCES[tmp_x_index, tmp_y_index] - DISTANCES[7, 58]) + np.square((tmp_x_index - 10)*2 + 4))
            self.x_index = tmp_x_index
            self.y_index = tmp_y_index
            self.flag = 5 < self.x_index < 9 and 54 < self.y_index < 62
            self.done = self.flag and action == 0
            obs = STATE_ARRAY[:, :, self.x_index, self.y_index, np.newaxis]
            reward = 0.1*(1 - self.done)*(-1 + 2*right_choice) + self.done
        self.nbEpisode = self.nbEpisode + 1*self.done

        return obs, reward
