import gym
import numpy as np
import random
import math
import cv2
import imageio
from gym import spaces
from matplotlib import pyplot as plt

PHI_MAX = 30

# Set here manually.
X_STATES = 400
Y_STATES = 400
X_TILT_STATES = 101
ROT_STATES = 101
NUM_OF_ACTIONS = 9

IN_DIM = (1, 84, 84)
MASK_FILE = '/content/drive/My Drive/UBC Research/mask.png'
INTERPOLATION = cv2.INTER_NEAREST

class navigate2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, path, is_test=0, is_same=0):
        self.state_array = np.load(path)  # Convert .mat file to .npy !!
        self.state_array = self.state_array.transpose(0, 2, 1)  # Revise this part after having the method
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
        self.y_index = random.randint(0, Y_STATES - 1)
        self.x_tilt_index = random.randint(0, X_TILT_STATES - 1)
        self.rot_index = random.randint(0, ROT_STATES - 1)

        self.mask = imageio.imread(MASK_FILE)[:, :, 1]
        self.mask = np.uint8(self.mask/255)
        self.mask = cv2.resize(self.mask, dsize=(400, 400), interpolation=INTERPOLATION)
         
        #  Revise this part after having the method
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
        self.state_array = self.data + random.randint(0, 50)*np.random.randn(400, 400, 400)  # Revise the array dimension after having new data
        self.state_array = np.clip(self.state_array, 0, 255)
        self.state_array = np.array(self.state_array, np.uint8)

        self.x_index = random.randint(0, X_STATES - 1)
        self.y_index = random.randint(0, Y_STATES - 1)
        self.x_tilt_index = random.randint(0, X_TILT_STATES - 1)
        self.rot_index = random.randint(0, ROT_STATES - 1)

        self.state = self.get_slice(self.rot_index*2/(ROT_STATES - 1) - 1, self.tilt_index*2/(TILT_STATES - 1) - 1, self.x_index*2/(X_STATES - 1) - 1)  # Revise this part after having the method
        state = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)
        state = state[np.newaxis, :, :]  # Check if the new method also returns image in (H, W) format

        return state

    def render(self, mode='human'):
        plt.imshow(self.state, cmap='gray', vmin=0, vmax=255)
        plt.show()
        cv2.destroyAllWindows()
        print('(ROT, TILT_X, X, Y): (' + str(self.rot_index) + ', ' + str(self.x_tilt_index) + ', ' + str(self.y_tilt_index) + ', ' + str(self.x_index) + ', ' + str(self.y_index) + ')')

    def take_action(self, action):

        temp_x = int((self.x_index - 8)*(action == 1) + (self.x_index + 8)*(action == 2) + self.x_index*(action != 1 and action != 2))
        temp_y = int((self.y_index - 8)*(action == 3) + (self.y_index + 8)*(action == 4) + self.y_index*(action != 3 and action != 4))
        temp_x_tilt = int((self.x_tilt_index - 2)*(action == 5) + (self.x_tilt_index + 2)*(action == 6) + self.x_tilt_index*(action != 5 and action != 6))      
        temp_rot = int((self.rot_index - 2)*(action == 7) + (self.rot_index + 2)*(action == 8) + self.rot_index*(action != 7 and action != 8))

        if temp_x < 0 or temp_x > (X_STATES - 1) or temp_y < 0 or temp_y > (Y_STATES - 1) or\
                temp_x_tilt < 0 or temp_x_tilt > (X_TILT_STATES - 1) or\
                temp_rot < 0 or temp_rot > (ROT_STATES - 1):
            obs = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)[np.newaxis, :, :]  # Check here if method returns (H, W) format
            reward = -0.1

        # Check the target image intervals
        else:
            reinf = np.abs(self.x_index - 199) + np.abs(self.y_index - 199) + np.abs(self.x_tilt_index - 50) + np.abs(self.rot_index - 50)\
                    > np.abs(temp_x - 199) + np.abs(temp_y - 199) + np.abs(temp_x_tilt - 50) + np.abs(temp_rot - 50)
            self.x_index = temp_x
            self.y_index = temp_y
            self.x_tilt_index = temp_x_tilt
            self.rot_index = temp_rot
            self.flag = 195 < self.x_index < 204 and 195 < self.y_index < 204 and 48 < self.x_tilt_index < 51 and 48 < self.rot_index < 51
            self.done = self.flag and action == 0
            self.state = self.get_slice(self.rot_index*2/(ROT_STATES - 1) - 1, self.tilt_index*2/(TILT_STATES - 1) - 1, self.x_index*2/(X_STATES - 1) - 1)  #  New method
            obs = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)[np.newaxis, :, :]  # Again check if new method returns (H, W)
            reward = 0.1*(1 - self.done)*(-1 + 2*reinf) + self.done

        self.nbEpisode = self.nbEpisode + 1*self.done

        return obs, reward
