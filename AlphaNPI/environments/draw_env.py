import os
import sys
import time
import math
import cmath
from PIL import Image
from skimage.draw import line, circle_perimeter
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from environments.environment import Environment


class DrawEnvEncoder(nn.Module):
    '''
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    '''

    def __init__(self, observation_dim, encoding_dim):
        super(DrawEnvEncoder, self).__init__()
        channels = [1, 16, 32, 64]
        self.conv1 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[1], channels[2], 3, padding=1)
        self.conv3 = nn.Conv2d(channels[2], channels[3], 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.ln1 = nn.Linear(9216, encoding_dim)

        self = self.cuda()

    def forward(self, x):
        #We need to resphape the input because it is being passed in flat because the other programs wouldn't need convolutions
        x = x.view(-1,1,100,100)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x,start_dim=1)
        x = F.relu(self.ln1(x))
        return x


class DrawEnv(Environment):
    """Class that represents a list environment. It represents a list of size length of digits. The digits are 10-hot-encoded.
    There are two pointers, each one pointing on a list element. Both pointers can point on the same element.

    The environment state is composed of a scratchpad of size length x 10 which contains the list elements encodings
    and of the two pointers positions.

    An observation is composed of the two encoding of the elements at both pointers positions.

    Primary actions may be called to move pointers and swap elements at their positions.

    We call episode the sequence of actions applied to the environment and their corresponding states.
    The episode stops when the list is sorted.
    """

    def __init__(self, dim=100, encoding_dim=32, hierarchy=True):

        assert dim > 0, "length must be a positive integer"
        self.dim = dim
        #Added this in case we think we need a non square env
        self.width = dim
        self.height = dim
        self.current_canvas = np.array(self._create_new_canvas())

        # self.current_pixel_data = np.array(self.current_canvas)

        # self.current_pixel_data = self.current_canvas.load()
        self.current_pix = np.array([dim//2]*2)
        self.stride = 4
        self.encoding_dim = encoding_dim
        self.has_been_reset = False
        self.unit = 8

        if hierarchy:
            self.programs_library = {'CZ_STOP': {'level': -1, 'recursive': False, "continuous":False, "crange":None},
                                     'C_MOVE': {'level': 0, 'recursive': False, "continuous":True, "crange":[0,2*np.pi]},
                                     'D_ULINE': {'level': 1, 'recursive': False, "continuous":False, "crange":None},
                                     'D_DLINE': {'level': 1, 'recursive': False, "continuous":False, "crange":None},
                                     'D_LLINE': {'level': 1, 'recursive': False, "continuous":False, "crange":None},
                                     'D_RLINE': {'level': 1, 'recursive': False, "continuous":False, "crange":None},
                                    #  'D_CIRCLE': {'level': 1, 'recursive': False, "continuous":False, "crange":None},
                                     # 'TRIANGLE': {'level': 2, 'recursive': False, "continuous":False, "crange":None},
                                     # 'LSHAPE': {'level': 2, 'recursive': False, "continuous":False, "crange":None},
                                     # 'SQUARE': {'level': 3, 'recursive': False, "continuous":False, "crange":None}
                                     }
            for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
                self.programs_library[key]['index'] = idx

            self.move_angles = {'C_UMOVE': 0.5*np.pi,
                                'C_DMOVE': 1.5*np.pi,
                                'C_LMOVE': 1*np.pi,
                                'C_RMOVE': 0.0}

            self.prog_to_func = {'CZ_STOP': self._stop,
                                 'C_MOVE': self._move
                                }

            self.prog_to_precondition = {'CZ_STOP': self._stop_precondition,
                                         'C_MOVE': self._move_precondition,
                                         'D_ULINE': self._line_precondition,
                                         'D_DLINE': self._line_precondition,
                                         'D_LLINE': self._line_precondition,
                                         'D_RLINE': self._line_precondition,
                                         'D_CIRCLE': self._circle_precondition,
                                         # 'TRIANGLE': self._shape_precondition,
                                         # 'LSHAPE': self._shape_precondition,
                                         # 'SQUARE': self._shape_precondition
                                         }

            square_vertices = [(100,100), (50,100), (50, 50), (100, 50), (100,100)]
            triangle_vertices = [(100,100), (125,75), (150, 100), (100,100)]

            self.line_directions = {'D_ULINE': [-self.unit, 0],
                                    'D_DLINE': [self.unit, 0],
                                    'D_LLINE': [0, -self.unit],
                                    'D_RLINE': [0, self.unit]
                                   }

            self.prog_to_postcondition = {'D_ULINE': self._line_postcondition([-self.unit, 0]),
                                          'D_DLINE': self._line_postcondition([self.unit, 0]),
                                          'D_LLINE': self._line_postcondition([0, -self.unit]),
                                          'D_RLINE': self._line_postcondition([0, self.unit]),
                                          'D_CIRCLE': self._circle_postcondition,
                                          # 'TRIANGLE': self._shape_postcondition(triangle_vertices),
                                          # 'LSHAPE': self._shape_postcondition('LSHAPE'), #TODO
                                          # 'SQUARE': self._shape_postcondition(square_vertices)
                                         }

        else:
            # In no hierarchy mode, the only non-zero program is Bubblesort

            self.programs_library = {'STOP': {'level': -1, 'recursive': False},
                                     'MOVE': {'level': 0, 'recursive': False},
                                     
                                     'BUBBLESORT': {'level': 1, 'recursive': False}}
            for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
                self.programs_library[key]['index'] = idx

            self.prog_to_func = {'STOP': self._stop,
                                 'UMOVE': self._move(0.5*np.pi),
                                 'DMOVE': self._move(1.5*np.pi),
                                 'LMOVE': self._move(1*np.pi),
                                 'RMOVE': self._move(0)}

            self.prog_to_precondition = {'STOP': self._stop_precondition,
                                         'MOVE': self._move_precondition,
                                        }

            # self.prog_to_postcondition = {'FIGURE': self._figure_postcondition}

        super(DrawEnv, self).__init__(self.programs_library, self.prog_to_func,
                                               self.prog_to_precondition, self.prog_to_postcondition)

    def get_continuous_from_index(self, index):
        program = self.get_program_from_index(index)
        return self.programs_library[program]['continuous']


    def _stop(self):
        """Do nothing. The stop action does not modify the environment."""
        pass

    def _stop_precondition(self):
        return True


    def _move(self, action):
        #This finds the target pixel to move to
        cartesian = cmath.rect(self.stride, action)
        movement = np.array([cartesian.imag, cartesian.real])
        target = self.current_pix + movement.astype(int)

        rr, cc = line(self.current_pix[0], self.current_pix[1], target[0], target[1])
        self.current_canvas[rr, cc] = 1.0
        
        self.current_pix = target

    def _move_precondition(self):
        return True

    def _line_precondition(self):
        return True

    def _line_postcondition(self, direction):
        def _line(init_state, state):
            init_canvas, init_position = init_state
            canvas, position = state

            final_position = init_position + direction

            drawn_canvas = np.copy(init_canvas)
            rr, cc = line(init_position[0], init_position[1], final_position[0], final_position[1])
            drawn_canvas[rr, cc] = 1.0
            
            return np.equal(drawn_canvas, canvas).all() and np.equal(position, init_position + direction).all(), drawn_canvas, final_position
        
        return _line

    def _circle_precondition(self):
        return True

    def _circle_postcondition(self, init_state, state):
        init_canvas, init_position = init_state
        canvas, position = state

        drawn_canvas = np.copy(init_canvas)
        rr, cc = circle_perimeter(init_position[0], init_position[1] + self.unit//2, self.unit//2)
        drawn_canvas[rr, cc] = 1.0

        return np.equal(drawn_canvas, canvas).all() and np.equal(position, init_position).all(), drawn_canvas

    def _shape_precondition(self):
        return True

    def _shape_postcondition(self, vertices):
        def _shape(init_state, state):
            init_canvas, init_position = init_state
            canvas, position = state

            drawn_canvas = np.copy(init_canvas)
            for i, vertex in enumerate(vertices[1:], start=1):
                rr, cc = line(vertices[i-1][0], vertices[i-1][1], vertex[0], vertex[1])
                drawn_canvas[rr, cc] = 1.0
            return np.equal(drawn_canvas, canvas) and np.equal(position, init_position), drawn_canvas

        return _shape
    

    def _one_hot_encode(self, digit, basis=10):
        """One hot encode a digit with basis.

        Args:
          digit: a digit (integer between 0 and 9)
          basis:  (Default value = 10)

        Returns:
          a numpy array representing the 10-hot-encoding of the digit

        """
        encoding = np.zeros(basis)
        encoding[digit] = 1
        return encoding

    def _one_hot_decode(self, one_encoding):
        """Returns digit associated to a one hot encoding.

        Args:
          one_encoding: numpy array representing the 10-hot-encoding of a digit.

        Returns:
          the digit encoded in one_encoding

        """
        return np.argmax(one_encoding)

    def _create_new_canvas(self):
        return Image.new('1', (self.width, self.height), 0)

    def reset_env(self):
        """Reset the environment. The list are values are draw randomly. The pointers are initialized at position 0
        (at left position of the list).

        """
        # start with an empty black canvas
        self.current_canvas = np.array(self._create_new_canvas()).astype(float)
        # start at center
        self.current_pix = np.array([self.width // 2, self.height // 2])
        cur_x, cur_y = self.current_pix.astype(int)
        self.current_canvas[cur_x, cur_y] = 1.0
        self.has_been_reset = True
        return self.get_observation()


    def get_state(self):
        """Returns the current state.

        Returns:
            the environment state

        """
        assert self.has_been_reset, 'Need to reset the environment before getting states'
        return np.copy(self.current_canvas), self.current_pix

    def get_observation(self):
        """Returns an observation of the current state.

        Returns:
            an observation of the current state
        """
        return np.array(self.current_canvas, dtype=np.float)


    def get_observation_dim(self):
        """

        Returns:
            the size of the observation tensor
        """
        return self.dim ** 2

    def _create_new_canvas(self):
        return Image.new('1', (self.width, self.height), 0)

    def reset_to_state(self, state):
        """

        Args:
          state: a given state of the environment
        reset the environment is the given state

        """
        self.current_canvas = state[0].copy()
        self.current_pix = state[1]

    def get_state_str(self, state):
        """Print a graphical representation of the environment state"""
        current_canvas = state[0].copy()  # check
        current_pix = state[1]
        out = 'canvas: {}, current_pix : {}'.format(str(current_canvas), current_pix)
        return out

    def compare_state(self, state1, state2):
        """
        Compares two states.

        Args:
            state1: a state
            state2: a state

        Returns:
            True if both states are equals, False otherwise.

        """
        bool = True
        bool &= np.array_equal(state1[0].load(), state2[0].load())
        bool &= (state1[1] == state2[1])
        return bool


    def get_line_reward(self):
        current_task = self.get_program_from_index(self.current_task_index)
        target_direction = self.line_directions[current_task]

        canvas, location = self.get_state()
        current_direction = location - np.array([self.dim//2, self.dim//2])

        _, target_angle = cmath.polar(complex(*target_direction))
        _, current_angle = cmath.polar(complex(*current_direction))

        return 50.0*(math.cos(current_angle-target_angle) + 1)


    # def get_reward(self):
    #     """Returns a reward for the current task at hand.
    #     Returns:
    #         Score based on how close the drawn image is to the target image.
    #     """
    #     task_init_state = self.tasks_dict[len(self.tasks_list)]
    #     canvas, location = self.get_state()
    #     current_task = self.get_program_from_index(self.current_task_index)
    #     # This should return the canvas I want
    #     post_program = self.prog_to_postcondition[current_task]
    #     done, target_canvas = post_program(task_init_state, self.get_state())
    #     # gaussian_canvas = gaussian_filter(target_canvas, sigma=3)
        
    #     intersection = np.multiply(canvas, target_canvas)
    #     union = np.logical_or(target_canvas, canvas)
    #     score = np.sum(intersection) 
    #     return score

    def get_reward(self):
        task_init_state = self.tasks_dict[len(self.tasks_list)]
        canvas, location = self.get_state()
        current_task = self.get_program_from_index(self.current_task_index)
        # This should return the canvas I want
        post_program = self.prog_to_postcondition[current_task]
        done, target_canvas, final_position = post_program(task_init_state, self.get_state())

        return -distance.euclidean(location, final_position)

