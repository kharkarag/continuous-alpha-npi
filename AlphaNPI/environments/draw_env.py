import sys
import time
import math
from PIL import Image

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
        channels = [2, 10, 30]
        self.conv1 = nn.Conv2d(channels[0], channels[1], 3, padding=1, dilation=0)
        self.conv2 = nn.Conv2d(channels[1], channels[2], 3, padding=1, dilation=0)
        self.conv3 = nn.Conv2d(channels[2], encoding_dim, 3, padding=1, dilation=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
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

    def __init__(self, dim=200, encoding_dim=32, hierarchy=True):

        assert dim > 0, "length must be a positive integer"
        self.dim = dim
        self.current_canvas = self._create_new_canvas()
        self.current_pixel_data = self.current_canvas.load()
        self.current_pix = np.array([dim/2 + 0.5]*2)
        self.encoding_dim = encoding_dim
        self.has_been_reset = False

        if hierarchy:
            self.programs_library = {'STOP': {'level': -1, 'recursive': False},
                                     'MOVE': {'level': 0, 'recursive': False},
                                     'LINE': {'level': 1, 'recursive': False},
                                     'CIRCLE': {'level': 1, 'recursive': False},
                                     'TRIANGLE': {'level': 2, 'recursive': False},
                                     'LSHAPE': {'level': 2, 'recursive': False},
                                     'SQUARE': {'level': 3, 'recursive': False}
                                     }
            for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
                self.programs_library[key]['index'] = idx

            self.prog_to_func = {'STOP': self._stop,
                                 'MOVE': self._move}

            self.prog_to_precondition = {'STOP': self._stop_precondition,
                                         'MOVE': self._move_precondition,
                                        }

            self.prog_to_postcondition = {'LINE': self._line_postcondition,
                                          'CIRCLE': self._circle_postcondition,
                                          'TRIANGLE': self._triangle_postcondition,
                                          'LSHAPE': self._lshape_postcondition,
                                          'SQUARE': self._square_postcondition
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

            self.prog_to_postcondition = {'BUBBLESORT': self._bubblesort_postcondition}

        super(DrawEnv, self).__init__(self.programs_library, self.prog_to_func,
                                               self.prog_to_precondition, self.prog_to_postcondition)


    def _stop(self):
        """Do nothing. The stop action does not modify the environment."""
        pass

    def _stop_precondition(self):
        return True


    def _move(self):
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
        self.current_canvas = self._create_new_canvas()
        self.current_pixel_data = self.current_canvas.load()
        # start at center
        self.current_pos = (self.width // 2 + 0.5, self.height // 2 + 0.5)
        return self._observe()
        self.has_been_reset = True

    def get_state(self):
        """Returns the current state.

        Returns:
            the environment state

        """
        assert self.has_been_reset, 'Need to reset the environment before getting states'
        return self.current_canvas.copy(), self.current_pix

    def get_observation(self):
        """Returns an observation of the current state.

        Returns:
            an observation of the current state
        """
        return np.array(self.current_pixel_data, dtype=np.float)/255.0

    def get_observation_dim(self):
        """

        Returns:
            the size of the observation tensor
        """
        return self.dim ** 2

    def reset_to_state(self, state):
        """

        Args:
          state: a given state of the environment
        reset the environment is the given state

        """
        self.current_canvas = state[0].copy()
        self.current_pixel_data = self.current_canvas.load()
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
