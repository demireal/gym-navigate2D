import gym
import numpy as np
import math
import cv2
import imageio
import scipy.io as spi
from gym import spaces
from matplotlib import pyplot as plt
from random import randint, choice

PHI_MAX = 30
THETA_SCALE = 0.15
XY_SCALE = 0.85
MASKSIZE = 400

# Set here manually.
X_STATES = 201
Y_STATES = 201
X_TILT_STATES = 101
ROT_STATES = 101
NUM_OF_ACTIONS = 9

IN_DIM = (1, 50, 50)
MASK_FILE = '/content/drive/My Drive/UBC Research/Data/mask.png'
INTERPOLATION = cv2.INTER_NEAREST

class navigate2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, path, fname, is_test=0, is_same=0):
        self.state_array = spi.loadmat(path)[fname]  
        self.state_array = np.swapaxes(self.state_array, 1, 2)  
        self.data = self.state_array
        self.is_test = is_test
        self.is_same = is_same
        self.nbEpisode = 1
        self.action_space = spaces.Discrete(NUM_OF_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=IN_DIM, dtype='uint8')
        self.flag = False
        self.done = False
        self.state = None
        self.x_index = randint(0, X_STATES - 1)
        self.y_index = randint(0, Y_STATES - 1)
        self.x_tilt_index = randint(0, X_TILT_STATES - 1)
        self.rot_index = randint(0, ROT_STATES - 1)

        self.mask = imageio.imread(MASK_FILE)[:, :, 1]
        self.mask = np.uint8(self.mask/255)
         
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
        self.state_array = self.data 
        self.state_array = np.clip(self.state_array, 0, 255)
        self.state_array = np.array(self.state_array, np.uint8)
        
        if self.nbEpisode < 50:
            self.x_index = randint(0, X_STATES - 1)
            self.y_index = randint(0, Y_STATES - 1)
            self.x_tilt_index = randint(0, X_TILT_STATES - 1)
            self.rot_index = randint(0, ROT_STATES - 1)

        else:
            self.x_index = randint(*choice([(0, (X_STATES - 1)/20), ((X_STATES - 1)/20, (X_STATES - 1) - (X_STATES - 1)/20), ((X_STATES - 1) - (X_STATES - 1)/20, (X_STATES - 1))]))
            self.y_index = randint(*choice([(0, (Y_STATES - 1)/20), ((Y_STATES - 1)/20, (Y_STATES - 1) - (Y_STATES - 1)/20), ((Y_STATES - 1) - (Y_STATES - 1)/20, (Y_STATES - 1))]))
            self.x_tilt_index = randint(*choice([(0, (X_TILT_STATES - 1)/20), ((X_TILT_STATES - 1)/20, (X_TILT_STATES - 1) - (X_TILT_STATES - 1)/20), ((X_TILT_STATES - 1) - (X_TILT_STATES - 1)/20, (X_TILT_STATES - 1))]))
            self.rot_index = randint(*choice([(0, (ROT_STATES - 1)/20), ((ROT_STATES - 1)/20, (ROT_STATES - 1) - (ROT_STATES - 1)/20), ((ROT_STATES - 1) - (ROT_STATES - 1)/20, (ROT_STATES - 1))]))

        self.state = self.get_slice(self.rot_index*2/(ROT_STATES - 1) - 1, self.x_tilt_index*2/(X_TILT_STATES - 1) - 1, self.x_index*2/(X_STATES - 1) - 1, self.y_index*2/(Y_STATES - 1) - 1)
        state = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)
        state = state[np.newaxis, :, :]

        return state

    def render(self, mode='human'):
        plt.imshow(self.state, cmap='gray', vmin=0, vmax=255)
        plt.show()
        cv2.destroyAllWindows()
        print('(ROT, TILT_X, X, Y): (' + str(self.rot_index) + ', ' + str(self.x_tilt_index) + ', ' + str(self.x_index) + ', ' + str(self.y_index) + ')')

    def take_action(self, action):

        temp_x = int((self.x_index - 2)*(action == 1) + (self.x_index + 2)*(action == 2) + self.x_index*(action != 1 and action != 2))
        temp_y = int((self.y_index - 2)*(action == 3) + (self.y_index + 2)*(action == 4) + self.y_index*(action != 3 and action != 4))
        temp_x_tilt = int((self.x_tilt_index - 1)*(action == 5) + (self.x_tilt_index + 1)*(action == 6) + self.x_tilt_index*(action != 5 and action != 6))      
        temp_rot = int((self.rot_index - 1)*(action == 7) + (self.rot_index + 1)*(action == 8) + self.rot_index*(action != 7 and action != 8))

        if temp_x < 0 or temp_x > (X_STATES - 1) or temp_y < 0 or temp_y > (Y_STATES - 1) or\
                temp_x_tilt < 0 or temp_x_tilt > (X_TILT_STATES - 1) or\
                temp_rot < 0 or temp_rot > (ROT_STATES - 1):
            obs = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)[np.newaxis, :, :]
            reward = -0.1

        else:
            reinf = np.abs(self.x_index - 100) + np.abs(self.y_index - 100) + np.abs(self.x_tilt_index - 50) + np.abs(self.rot_index - 50)\
                    > np.abs(temp_x - 100) + np.abs(temp_y - 100) + np.abs(temp_x_tilt - 50) + np.abs(temp_rot - 50)
            self.x_index = temp_x
            self.y_index = temp_y
            self.x_tilt_index = temp_x_tilt
            self.rot_index = temp_rot
            self.flag = 97 < self.x_index < 103 and 97 < self.y_index < 103 and 48 < self.x_tilt_index < 52 and 48 < self.rot_index < 52
            self.done = self.flag and action == 0
            self.state = self.get_slice(self.rot_index*2/(ROT_STATES - 1) - 1, self.x_tilt_index*2/(X_TILT_STATES - 1) - 1, self.x_index*2/(X_STATES - 1) - 1, self.y_index*2/(Y_STATES - 1) - 1)
            obs = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)[np.newaxis, :, :]
            reward = 0.1*(1 - self.done)*(-1 + 2*reinf) + self.done

        self.nbEpisode = self.nbEpisode + 1*self.done

        return obs, reward
    
    def get_bounding_box(self, theta, phi, dx, dy):
        h1 = [self.x0 / 2 + self.y0 / 2 * math.sin(theta) + dx,#+ dist,##*math.cos(theta),
              self.y0 / 2 - self.y0 / 2 * math.cos(theta) + dy]## + dist*math.sin(theta)]

        h2 = [self.x0 / 2 - self.y0 / 2 * math.sin(theta) + dx,#dist,##*math.cos(theta),
              self.y0 / 2 + self.y0 / 2 * math.cos(theta) + dy]## + dist*math.sin(theta)]

        z_min = 0 #self.z0 / 2 - self.z0 / 2 * math.cos(phi)
        z_max = self.z0 * math.cos(phi) #self.z0 / 2 + self.z0 / 2 * math.cos(phi)
        return h1, h2, z_min, z_max

    def get_slice(self, theta_n, phi_n, dx_n, dy_n):
        theta = theta_n*math.pi*THETA_SCALE
        phi = math.radians(phi_n*PHI_MAX)
        dx = dx_n*XY_SCALE*self.x0/2 # +/- 200 pixels
        dy = dy_n*XY_SCALE*self.y0/2 # +/- 350 pixels

        # --- 1: Get bounding box dims ---
        h1, h2, z_min, z_max = self.get_bounding_box(theta=theta, phi=phi, dx=dx, dy=dy)
        w = MASKSIZE
        h = MASKSIZE

        # --- 2: Extract slice from volume ---
        # Get x_i and y_i for current layer
        # TODO: check that replacing h with zmax is correct
        x_offsets = np.linspace(z_min, z_max, h) * math.sin(phi) * math.cos(theta) #np.linspace(-h/2, h/2, h) * math.sin(phi) * math.cos(theta)
        y_offsets = np.linspace(z_min, z_max, h) * math.sin(phi) * math.sin(theta) #np.linspace(-h/2, h/2, h) * math.sin(phi) * math.sin(theta)

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

