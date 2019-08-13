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
THETA_MAX = 45
X_SCALE = 0.5
Y_SCALE = 0.5
MASKSIZE = 400
DOWNSIZE_FACTOR = 4     # must either be 2, 4 or 8

# Set here manually.
X_STATES = 101
Y_STATES = 91
X_TILT_STATES = 101
ROT_STATES = 101
NUM_OF_ACTIONS = 9

IN_DIM = (1, 84, 84)
INFILE = '/content/drive/My Drive/UBC Research/Data/baby_data/downsized_mats2_zeroed.mat'
MASKFOLDER = '/content/drive/My Drive/UBC Research/Data/baby_data'
INTERPOLATION = cv2.INTER_NEAREST

class navigate2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, is_test=0):
        self.data = spi.loadmat(INFILE)['spliced_'+str(DOWNSIZE_FACTOR)+'x']
        self.data_orig = self.data
        
        self.mask_size = int(MASKSIZE / DOWNSIZE_FACTOR)
        self.mask = imageio.imread(MASKFOLDER+'/mask_'+str(DOWNSIZE_FACTOR)+'x.png')[:,:,1]
        self.mask = np.uint8(self.mask/255)
         
        dims = self.data.shape
        self.x0 = dims[0]
        self.y0 = dims[1]
        self.z0 = dims[2]
        
        self.flag = False
        self.done = False
        self.state = None
        self.x_index = randint(0, X_STATES - 1)
        self.y_index = randint(0, Y_STATES - 1)
        self.x_tilt_index = randint(0, X_TILT_STATES - 1)
        self.rot_index = randint(0, ROT_STATES - 1)
        
        self.action_space = spaces.Discrete(NUM_OF_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=IN_DIM, dtype='uint8')
        self.is_test = is_test
        self.nbEpisode = 1

    def step(self, action):
        state, reward = self.take_action(action)
        return state, reward, self.done, {}

    def reset(self):
        self.flag = False
        self.done = False
        self.data = self.data_orig + randint(0, 20)*np.random.randn(self.x0, self.y0, self.z0)
        
        if self.nbEpisode < 1000:
            self.x_index = randint(0, X_STATES - 1)
            self.y_index = randint(0, Y_STATES - 1)
            self.x_tilt_index = randint(0, X_TILT_STATES - 1)
            self.rot_index = randint(0, ROT_STATES - 1)
        
        else:
            self.x_index = randint(*choice([(0, 5), (6, 113), (114, 119)]))
            self.y_index = randint(*choice([(0, 4), (5, 102), (103, 107)]))
            self.x_tilt_index = randint(*choice([(0, 4), (5, 95), (96, 100)]))
            self.rot_index = randint(*choice([(0, 4), (5, 95), (96, 100)]))

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
        temp_x_tilt = int((self.x_tilt_index - 2)*(action == 5) + (self.x_tilt_index + 2)*(action == 6) + self.x_tilt_index*(action != 5 and action != 6))      
        temp_rot = int((self.rot_index - 2)*(action == 7) + (self.rot_index + 2)*(action == 8) + self.rot_index*(action != 7 and action != 8))

        if temp_x < 0 or temp_x > (X_STATES - 1) or temp_y < 0 or temp_y > (Y_STATES - 1) or\
                temp_x_tilt < 0 or temp_x_tilt > (X_TILT_STATES - 1) or\
                temp_rot < 0 or temp_rot > (ROT_STATES - 1):
            obs = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)[np.newaxis, :, :]
            reward = -0.1

        else:
            reinf = np.abs(self.x_index - 50) + np.abs(self.y_index - 45) + np.abs(self.x_tilt_index - 50) + np.abs(self.rot_index - 50)\
                    > np.abs(temp_x - 50) + np.abs(temp_y - 45) + np.abs(temp_x_tilt - 50) + np.abs(temp_rot - 50)
            self.x_index = temp_x
            self.y_index = temp_y
            self.x_tilt_index = temp_x_tilt
            self.rot_index = temp_rot
            self.flag = 48 < self.x_index < 52 and  43 < self.y_index < 47 and 48 < self.x_tilt_index < 52 and 48 < self.rot_index < 52
            self.done = self.flag and action == 0
            self.state = self.get_slice(self.rot_index*2/(ROT_STATES - 1) - 1, self.x_tilt_index*2/(X_TILT_STATES - 1) - 1, self.x_index*2/(X_STATES - 1) - 1, self.y_index*2/(Y_STATES - 1) - 1)
            obs = cv2.resize(self.state, dsize=(IN_DIM[1], IN_DIM[2]), interpolation=INTERPOLATION)[np.newaxis, :, :]
            reward = 0.1*(1 - self.done)*(-1 + 2*reinf) + self.done

        self.nbEpisode = self.nbEpisode + 1*self.done

        return obs, reward
    
    def get_bounding_box(self, theta, phi, dx, dy):
        #print('theta:',theta,'\tphi:',phi,'\tdx:',dx,'\tdy:',dy)
        h1 = [self.x0 / 2 - self.mask_size / 2 * math.sin(theta) + dx,#+ dist,##*math.cos(theta),
              self.y0 / 2 + self.mask_size / 2 * math.cos(theta) + dy]## + dist*math.sin(theta)]

        h2 = [self.x0 / 2 + self.mask_size / 2 * math.sin(theta) + dx,#dist,##*math.cos(theta),
              self.y0 / 2 - self.mask_size / 2 * math.cos(theta) + dy]## + dist*math.sin(theta)]

        z_min = 0 #self.z0 / 2 - self.z0 / 2 * math.cos(phi)
        z_max = self.z0 * math.cos(phi) #self.z0 / 2 + self.z0 / 2 * math.cos(phi)
        #print('h1:',h1,'\th2:',h2,'\tz_min:', z_min,'\tz_max:', z_max)
        return h1, h2, z_min, z_max

    def get_slice(self, theta_n, phi_n, dx_n, dy_n):
        theta = math.radians(theta_n*THETA_MAX)
        phi = math.radians(phi_n*PHI_MAX)
        dx = X_SCALE*dx_n*self.x0/2  # +/- 200 pixels
        dy = Y_SCALE*dy_n*self.y0/2  # +/- 350 pixels

        # --- 1: Get bounding box dims ---
        h1, h2, z_min, z_max = self.get_bounding_box(theta=theta, phi=phi, dx=dx, dy=dy)
        w = self.mask_size
        h = self.mask_size

        # --- 2: Extract slice from volume ---
        # Get x_i and y_i for current layer
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
        the_slice = np.take(self.data, flat_inds)

        # --- 3: Mask slice ---
        the_slice = np.multiply(the_slice, self.mask)
        return the_slice

