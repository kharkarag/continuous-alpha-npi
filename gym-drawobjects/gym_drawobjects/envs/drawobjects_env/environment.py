from io import StringIO
import sys
import time
import math
from PIL import Image

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

from .gui import GUI
from .labels import LABELS
from . import inception

def truncate(s):
    if len(s) > 10:
        return s[:7]+'...'
    else:
        return s

class DrawObjectsEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self):
        super(DrawObjectsEnv, self).__init__()
        self._is_rendering = False

        self.width = 200
        self.height = 200
        self.stride = 2
        self.label_idx = 1
        self.current_pix = np.array([100.5,100.5])

        self.last_reward = 0

        self._reset()

    def action_to_human_readable(self, action):

        return str(tuple(action))

    @property
    def observation_space(self):
        return spaces.Box(low=0.0, high=1.0, shape=(200, 200))

    @property
    def action_space(self):
        return spaces.Box(low=-10, high=10, shape=(2,), dtype=int)

    def _create_new_canvas(self):
        return Image.new('L', (self.width, self.height), 'white')

    def _init_gui(self):
        self.gui = GUI()
        self.gui.start()

    def _reset(self):
        inception.init()
        # start with an empty white canvas
        self.current_canvas = self._create_new_canvas()
        self.current_pixel_data = self.current_canvas.load()
        # start at center
        self.current_pos = (self.width // 2 + 0.5, self.height // 2 + 0.5)
        return self._observe()

    def _redraw(self):
        if not self._is_rendering:
            return
        self.gui.update(self.current_canvas, self.last_reward, truncate(LABELS[self.label_idx-1]))

    def _render(self, mode="human", close=True):
        if close: return
        if not self._is_rendering:
            self._is_rendering = True
            self._init_gui()

        self._redraw()
        return None
        
    def _apply_action(self, action):
        self.current_pixel_data[self.current_pos] = 0

        target = self.current_pos + 2*action.astype(int)
        current_pixel = (int(self.current_pos[0]), int(self.current_pos[1]))
        
        while (target < [self.width-2, self.height-2]).all() and (target > 1).all():
            nex = np.copy(self.current_pos)

            for i in range(len(self.current_pos)):
                if action[i] >= 0:
                    nex[i] = np.floor(self.current_pos[i] + np.sign(action[i]))
                else:
                    nex[i] = np.ceil(self.current_pos[i] + np.sign(action[i]))

            deltas = (nex - self.current_pos) / action
            min_idx = np.argmin(np.abs(deltas))

            maj_length = (deltas*action)[min_idx]
            min_length = action[1-min_idx]/action[min_idx] * maj_length

            if min_idx == 0:
                self.current_pos = (self.current_pos[0] + maj_length, self.current_pos[1] + min_length)
            else:
                self.current_pos = (self.current_pos[0] + min_length, self.current_pos[1] + maj_length)

            current_pixel = (int(self.current_pos[0]), int(self.current_pos[1]))
            self.current_pixel_data[current_pixel] = 0

            if (np.abs(self.current_pos - target) < [1,1]).all() \
            or (np.array(self.current_pos) > [self.width-2, self.height-2]).any() \
            or (np.array(self.current_pos) < 1).any():
                break

        self.current_pos = (current_pixel[0] + 0.5, current_pixel[1] + 0.5)

    def _observe(self):
        return np.array(self.current_canvas, dtype=np.float)/255.0
    
    def _reward(self, action):
        # self.last_reward = inception.get_prediction(self.current_canvas, self.label_idx)
        return self.last_reward

    def _step(self, action):
        self._apply_action(action)
        reward = self._reward(action)

        obs = self._observe()
        done = False
        info = {}

        return obs, reward, done, info

    def _seed(self, seed=0):
        return [seed]
