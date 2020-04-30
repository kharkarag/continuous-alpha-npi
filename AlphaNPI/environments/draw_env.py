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

REF_IMG_DIR = "ref_img"

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
        #Before Changes
        # channels = [1, 10, 30]
        # self.conv1 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        # self.conv2 = nn.Conv2d(channels[1], channels[2], 3, padding=1)
        # self.conv3 = nn.Conv2d(channels[2], encoding_dim, 3, padding=1)


    def forward(self, x):
        #We need to resphape the input because it is being passed in flat because the other programs wouldn't need convolutions
        # print(x.size())
        x = x.view(-1,1,100,100)

        x = x.cuda()
        # print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.size())
        x = torch.flatten(x,start_dim=1)
        # print(x.size())
        x = F.relu(self.ln1(x))
        # print(x)
        return x.cpu()

        # #Before Changes
        # #We need to resphape the input because it is being passed in flat because the other programs wouldn't need convolutions
        # x = x.view(-1,1,200,200)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = torch.sigmoid(self.conv3(x))
        # return x



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
        self.stride = 3
        self.encoding_dim = encoding_dim
        self.has_been_reset = False

        if hierarchy:
            self.programs_library = {'STOP': {'level': -1, 'recursive': False, "continuous":False, "crange":None},
                                     'MOVE': {'level': 0, 'recursive': False, "continuous":True, "crange":[0,2*np.pi]},
                                    #  'ULINE': {'level': 1, 'recursive': False, "continuous":False, "crange":None}
                                     # 'DLINE': {'level': 1, 'recursive': False, "continuous":False, "crange":None},
                                     'LLINE': {'level': 1, 'recursive': False, "continuous":False, "crange":[0,2*np.pi]},
                                     # 'RLINE': {'level': 1, 'recursive': False, "continuous":False, "crange":None},
                                     # 'CIRCLE': {'level': 1, 'recursive': False, "continuous":False, "crange":None},
                                     # 'TRIANGLE': {'level': 2, 'recursive': False, "continuous":False, "crange":None},
                                     # 'LSHAPE': {'level': 2, 'recursive': False, "continuous":False, "crange":None},
                                     # 'SQUARE': {'level': 3, 'recursive': False, "continuous":False, "crange":None}
                                     }
            for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
                self.programs_library[key]['index'] = idx

            self.prog_to_func = {'STOP': self._stop,
                                 'MOVE': self._move}

            self.prog_to_precondition = {'STOP': self._stop_precondition,
                                         'MOVE': self._move_precondition,
                                        #  'ULINE': self._line_precondition,
                                         # 'DLINE': self._line_precondition,
                                         'LLINE': self._line_precondition,
                                         # 'RLINE': self._line_precondition,
                                         # 'CIRCLE': self._circle_precondition,
                                         # 'TRIANGLE': self._shape_precondition,
                                         # 'LSHAPE': self._shape_precondition,
                                         # 'SQUARE': self._shape_precondition
                                         }

            square_vertices = [(100,100), (50,100), (50, 50), (100, 50), (100,100)]
            triangle_vertices = [(100,100), (125,75), (150, 100), (100,100)]

            self.prog_to_postcondition = {#'ULINE': self._line_postcondition([-15, 0]),
                                          # 'DLINE': self._line_postcondition([50, 0]),
                                          'LLINE': self._line_postcondition([0, -15]),
                                          # 'RLINE': self._line_postcondition([0, 50]),
                                          # 'CIRCLE': self._circle_postcondition,
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
                                 'MOVE': self._move}

            self.prog_to_precondition = {'STOP': self._stop_precondition,
                                         'MOVE': self._move_precondition,
                                        }

            self.prog_to_postcondition = {'FIGURE': self._figure_postcondition}

        super(DrawEnv, self).__init__(self.programs_library, self.prog_to_func,
                                               self.prog_to_precondition, self.prog_to_postcondition)


    def _stop(self):
        """Do nothing. The stop action does not modify the environment."""
        pass

    def _stop_precondition(self):
        return True

    def _move(self, action):
        #This finds the target pixel to move to
        # cartesian = cmath.rect(self.stride, action)
        # movement = np.array([cartesian.imag, cartesian.real])
        # target = self.current_pix + np.rint(movement)
        # if target[0]>=0.0 and target[0]<= self.dim and target[1]>=0.0 and target[1]<= self.dim:
        #     #This is probably an inefficient way to find the pixels the line moves through
        #     x_move =  np.linspace(self.current_pix[0], target[0], num = self.stride)
        #     y_move = np.linspace(self.current_pix[1], target[1], num = self.stride)
        #     for p in range(x_move.shape[0]):
        #         self.current_canvas[int(x_move[p]), int(y_move[p])] = 0.0
        #     self.current_pix = target

        #Find pixels line passes through
        # rr, cc = line(self.current_pix[0], self.current_pix[1], target[0], target[1])
        # self.current_pixel_data[rr, cc] = 0

        # self.current_pix = target

        cartesian = cmath.rect(self.stride, action)
        movement = np.array([cartesian.imag, cartesian.real])
        
        target = self.current_pix + movement.astype(int)
        rr, cc = line(self.current_pix[0], self.current_pix[1], target[0], target[1])
        self.current_canvas[rr, cc] = 0
        
        self.current_pix = target

        # print(f"target: {target}")

    def _move_precondition(self):
        return True

    def _line_precondition(self):
        return True

    def _line_postcondition(self, direction):
        def _line(init_state, state):

            # init_canvas, init_position = init_state
            # canvas, position = state
            # position = np.array(position)
            # init_position = np.array(init_position)
            # ar_direction = np.array(direction)
            # drawn_canvas = np.copy(init_canvas)

            # target = init_position + ar_direction
            # # This is probably an inefficient way to find the pixels the line moves through
            # x_move = np.linspace(init_position[0], target[0], num=int(np.linalg.norm(ar_direction)))
            # y_move = np.linspace(init_position[1], target[1], num=int(np.linalg.norm(ar_direction)))
            # for p in range(x_move.shape[0]):
            #     drawn_canvas[int(x_move[p]), int(y_move[p])] = 0.0

            # return np.all(np.equal(drawn_canvas, canvas)) and np.all(np.equal(position, init_position + ar_direction)) , drawn_canvas

            ######################################
            init_canvas, init_position = init_state
            canvas, position = state
            
            drawn_canvas = np.copy(init_canvas)
            rr, cc = line(init_position[0], init_position[1], init_position[0] + direction[0], init_position[1] + direction[1])
            drawn_canvas[rr, cc] = 0
            
            # print(f"Line: {init_position + direction}")

            return np.all(np.equal(drawn_canvas, canvas)) and np.all(np.equal(position, init_position + direction)), drawn_canvas
        
        return _line

    def _circle_precondition(self):
        return True

    def _circle_postcondition(self, init_state, state):
        init_canvas, init_position = init_state
        canvas, position = state

        drawn_canvas = np.copy(init_canvas)
        rr, cc = circle_perimeter(init_canvas[0], init_canvas[1], 25)
        drawn_canvas[rr, cc] = 0

        return np.equal(drawn_canvas, canvas) and np.equal(position, init_position)

    def _shape_precondition(self):
        return True

    def _shape_postcondition(self, vertices):
        def _shape( init_state, state):
            init_canvas, init_position = init_state
            canvas, position = state

            drawn_canvas = np.copy(init_canvas)
            for i, vertex in enumerate(vertices[1:], start=1):
                rr, cc = line(vertices[i-1][0], vertices[i-1][1], vertex[0], vertex[1])
                drawn_canvas[rr, cc] = 0
            return np.equal(drawn_canvas, canvas) and np.equal(position, init_position)

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
        return Image.new('L', (self.width, self.height), 'white')

    def reset_env(self):
        """Reset the environment. The list are values are draw randomly. The pointers are initialized at position 0
        (at left position of the list).

        """
        # start with an empty white canvas
        self.current_canvas = np.array(self._create_new_canvas())
        cur_x, cur_y = self.current_pix.astype(int)
        self.current_canvas[cur_x, cur_y] = 0
        # self.current_pixel_data = np.array(self.current_canvas)
        # start at center
        self.current_pix = np.array([self.width // 2, self.height // 2])
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
        return np.array(self.current_canvas, dtype=np.float)/255.0


    def get_observation_dim(self):
        """

        Returns:
            the size of the observation tensor
        """
        return self.dim ** 2

    def _create_new_canvas(self):
        return Image.new('L', (self.width, self.height), 'white')

    def reset_to_state(self, state):
        """

        Args:
          state: a given state of the environment
        reset the environment is the given state

        """
        # self.current_canvas = state[0].copy()
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




    def get_reward(self):
        """Returns a reward for the current task at hand.
        Returns:
            Score based on how close the drawn image is to the target image.
        """
        task_init_state = self.tasks_dict[len(self.tasks_list)]
        canvas, location = self.get_state()
        current_task = self.get_program_from_index(self.current_task_index)
        #This should return the canvas I want
        post_program = self.prog_to_postcondition[current_task]
        done, target_canvas = post_program(task_init_state,self.get_state())


        target_canvas = 1-target_canvas/255
        gaussian_canvas = gaussian_filter(target_canvas, sigma=5)

        # intersection = np.logical_and(target_canvas == 0, canvas == 0)
        gaussian_canvas = np.maximum(gaussian_canvas, target_canvas)

        intersection = np.multiply(1-canvas/255, gaussian_canvas)
        intersection = intersection / np.amax(intersection)
        
        # print(intersection[intersection > 0])
        # print(np.argwhere(intersection != 0))

        union = np.logical_or(target_canvas == 1, canvas == 0)

        # score = np.sum(intersection)/np.sum(union)
        score = 100 * np.sqrt(np.sum(intersection)/np.sum(union))

        # print(intersection[intersection > 0])
        # print(union[union > 0])

        return score
