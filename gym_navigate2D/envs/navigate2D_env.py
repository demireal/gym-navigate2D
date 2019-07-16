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

CROP_HEIGHT = 164
CROP_WIDTH = 250
INPUT_SHAPE = (50, 50, 1)
X_STATES = 20
Y_STATES = 113
NUM_OF_ACTIONS = 5
STATE_ARRAY = np.zeros((INPUT_SHAPE[0], INPUT_SHAPE[1], X_STATES, Y_STATES))
df=pd.read_csv('gdrive/My Drive/UBC Research/xy.csv')
distances = df.values

for x_index in range(X_STATES):
    PATH_NAME = '/content/drive/My Drive/UBC Research/my_test_data_2D/' + str(x_index) + '/*.png'
    for y_index, im_path in enumerate(sorted(glob.glob(PATH_NAME))):
        im = imageio.imread(im_path)
        im = im[im.shape[0] - CROP_HEIGHT - 32:im.shape[0] - 32, int(im.shape[1]/2 - CROP_WIDTH/2):int(im.shape[1]/2 + CROP_WIDTH/2), :-1]  # crop unnecessary black parts, lose alpha channel.
        img = Image.fromarray(im)
        img = img.resize((INPUT_SHAPE[0], INPUT_SHAPE[1])).convert('L')  # resize, convert to grey scale
        im = np.array(img)
        STATE_ARRAY[:, :, x_index, y_index] = im/255


class navigate2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
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
        if self.nbEpisode < 15:
            self.x_index = random.randint(4, 12)
            self.y_index = random.randint(45, 69)
        elif self.nbEpisode < 50:
            self.x_index = random.randint(3, 13)
            self.y_index = random.randint(40, 74)
        elif self.nbEpisode < 100:
            self.x_index = random.randint(2, 14)
            self.y_index = random.randint(30, 84)
        else:
            self.x_index = random.randint(0, self.x_index - 1)
            self.y_index = random.randint(0, self.y_index - 1)
        self.state = STATE_ARRAY[:, :, x_index, y_index, np.newaxis]
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
            tmp_y_index = np.argmin(np.abs(distances[self.x_index, self.y_index] - distances[tmp_x_index, :]))
        elif action == 2:  # Right
            tmp_x_index = self.x_index - 1
            tmp_y_index = np.argmin(np.abs(distances[self.x_index, self.y_index] - distances[tmp_x_index, :]))
        elif action == 3:  # Up
            tmp_x_index = self.x_index
            tmp_y_index = self.y_index + 4
        else:  # Down
            tmp_x_index = self.x_index
            tmp_y_index = self.y_index - 4

        if tmp_x_index < 0 or tmp_x_index > X_STATES - 1 or tmp_y_index < 0 or tmp_y_index > Y_STATES - 1:
            obs = self.state
            reward = -0.1
        else:
            self.x_index = tmp_x_index
            self.y_index = tmp_y_index
            self.done = 7 < self.x_index < 9 and 55 < self.y_index < 61
            obs = STATE_ARRAY[:, :, self.x_index, self.y_index, np.newaxis]
            reward = -0.1*(1 - self.done) + self.done

        self.nbEpisode = self.nbEpisode + 1*self.done

        return obs, reward
