import os
import sys
import time
import math
import cmath
from PIL import Image
from skimage.draw import line, circle_perimeter
from scipy.ndimage import gaussian_filter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from environments.environment import Environment
from environments.draw_env import DrawEnv

env = DrawEnv()
env.start_task(env.programs_library['ULINE']['index'])

# env.current_task_index = env.programs_library['LLINE']['index']

scaler = 0.8

for i in range(5):
    env._move(scaler * np.pi)

print(f"{scaler * np.pi}: {env.get_reward()}")