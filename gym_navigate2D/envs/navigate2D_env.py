import gym
import numpy as np
import random
import pandas as pd
from time import sleep
from IPython.display import clear_output
from gym import spaces
from matplotlib import pyplot as plt

# Each state is an image. State space is 2D.

NUM_OF_ACTIONS = 5

DF = pd.read_csv('/content/drive/My Drive/UBC Research/xy.csv')
DISTANCES = DF.values
for index_1, entry_array in enumerate(DISTANCES):
    for index_2, entry in enumerate(entry_array):
        DISTANCES [index_1, index_2] = float(entry.replace(',', '.'))

      
class navigate2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, path, is_test=0):
        self.state_array = np.load(path)
        self.x_states = self.state_array.shape[0]
        self.y_states = self.state_array.shape[1]
        self.input_shape = (self.state_array.shape[2], self.state_array.shape[3], 1)
        self.is_test = is_test
        self.x_index = random.randint(0, self.x_states - 1)
        self.y_index = random.randint(0, self.y_states - 1)
        self.state = self.state_array[self.x_index, self.y_index, np.newaxis, :, :]
        self.flag = False
        self.done = False
        self.nbEpisode = 1
        self.action_space = spaces.Discrete(NUM_OF_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.input_shape, dtype='numpy.uint8')

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
            self.x_index = random.randint(0, self.x_states - 1)
            self.y_index = random.randint(0, self.y_states - 1)
        self.state = self.state_array[self.x_index, self.y_index, np.newaxis, :, :]
        self.flag = False
        self.done = False
        return self.state

    def render(self, mode='human'):
        plt.imshow(self.state[0, :, :], cmap='gray', vmin=0, vmax=255)
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
            tmp_y_index = np.argmin(np.abs(DISTANCES[self.x_index, self.y_index] - DISTANCES[np.mod(tmp_x_index, self.x_states), :]))
        elif action == 2:  # Right
            tmp_x_index = self.x_index - 1
            tmp_y_index = np.argmin(np.abs(DISTANCES[self.x_index, self.y_index] - DISTANCES[np.mod(tmp_x_index, self.x_states), :]))
        elif action == 3:  # Up
            tmp_x_index = self.x_index
            tmp_y_index = self.y_index + 3
        else:  # Down
            tmp_x_index = self.x_index
            tmp_y_index = self.y_index - 3
              
        if tmp_x_index < 0 or tmp_x_index > self.x_states - 1 or tmp_y_index < 0 or tmp_y_index > self.y_states - 1:
            obs = self.state
            reward = -0.1
        else:
            right_choice = (np.square(DISTANCES[self.x_index, self.y_index] - DISTANCES[7, 58]) + np.square((self.x_index - 10)*2 + 4)) > (np.square(DISTANCES[tmp_x_index, tmp_y_index] - DISTANCES[7, 58]) + np.square((tmp_x_index - 10)*2 + 4))
            self.x_index = tmp_x_index
            self.y_index = tmp_y_index
            self.flag = 5 < self.x_index < 9 and 54 < self.y_index < 62
            self.done = self.flag and action == 0
            obs = self.state_array[self.x_index, self.y_index, np.newaxis, :, :]
            reward = 0.1*(1 - self.done)*(-1 + 2*right_choice) + self.done
        self.nbEpisode = self.nbEpisode + 1*self.done

        return obs, reward
