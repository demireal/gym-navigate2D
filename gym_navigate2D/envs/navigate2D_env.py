import gym
import imageio
import glob
import numpy as np
import random

from time import sleep
from IPython.display import clear_output
from gym import error, spaces, utils
from gym.utils import seeding
from PIL import Image
from matplotlib import pyplot as plt

## Each state is an image. State space is 1D.

PATH_NAME = '/content/drive/My Drive/my_test_data/*.png'
CROP_HEIGHT = 500
CROP_WIDTH = 500
INPUT_SHAPE = (50, 50, 1)
NUM_OF_STATES = 103
NUM_OF_ACTIONS = 3
STATE_ARRAY = np.zeros((INPUT_SHAPE[0], INPUT_SHAPE[1], NUM_OF_STATES))

for index, im_path in enumerate(sorted(glob.glob(PATH_NAME))):
    im = imageio.imread(im_path)
    im = im[im.shape[0] - CROP_HEIGHT:im.shape[0], int(im.shape[1]/2 - CROP_WIDTH/2):int(im.shape[1]/2 + CROP_WIDTH/2), :-1]  # crop unnecessary black parts, lose alpha channel.
    img = Image.fromarray(im)
    img = img.resize((INPUT_SHAPE[0], INPUT_SHAPE[1])).convert('L')  # resize, convert to grey scale
    im = np.array(img)
    STATE_ARRAY[:, :, index] = im/255


class navigate2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.index = random.randint(0, NUM_OF_STATES - 1)
        self.state = STATE_ARRAY[:, :, self.index:self.index + 1]
        self.done = False
        self.nbEpisode = 1
        self.action_space = spaces.Discrete(NUM_OF_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1, shape=INPUT_SHAPE, dtype='float32')

    def step(self, action):
        self.state, reward = self.take_action(action)
        return self.state, reward, self.done, {}

    def reset(self):
        print('Episode: ' + str(self.nbEpisode))
        self.index = random.randint(0, NUM_OF_STATES - 1)
        self.state = STATE_ARRAY[:, :, self.index:self.index+1]
        self.done = False
        return self.state

    def render(self, mode='human'):
        plt.imshow(255*self.state[:, :, 0], cmap='gray', vmin=0, vmax=255)
        plt.show()
        print('Index: ' + str(self.index))
        sleep(1)
        clear_output()

    def take_action(self, action):
        if action == 0:
            tmp_index = self.index
        elif action == 1:
            tmp_index = self.index + 1
        else:
            tmp_index = self.index - 1

        if tmp_index < 0 or tmp_index > NUM_OF_STATES - 1:
            obs = self.state
            reward = -0.1
        else:
            self.index = tmp_index
            self.done = 59 < self.index < 64
            obs = STATE_ARRAY[:, :, self.index:self.index + 1]
            reward = -0.1*(1 - self.done) + 3*self.done

        self.nbEpisode = self.nbEpisode + 1*self.done

        return obs, reward
