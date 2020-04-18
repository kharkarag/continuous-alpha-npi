import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from environments.environment import Environment


class SimpleDrawEnvEncoder(nn.Module):
    '''
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    '''

    def __init__(self, observation_dim, encoding_dim):
        super(SimpleDrawEnvEncoder, self).__init__()
        channels = [2, 10, 30]
        self.conv1 = nn.Conv2d(channels[0], channels[1], 3, padding=1, dilation=0)
        self.conv2 = nn.Conv2d(channels[1], channels[2], 3, padding=1, dilation=0)
        self.conv3 = nn.Conv2d(channels[2], encoding_dim, 3, padding=1, dilation=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


class SimpleDrawEnv(Environment):
    """Class that represents a list environment. It represents a list of size length of digits. The digits are 10-hot-encoded.
    There are two pointers, each one pointing on a list element. Both pointers can point on the same element.

    The environment state is composed of a scratchpad of size length x 10 which contains the list elements encodings
    and of the two pointers positions.

    An observation is composed of the two encoding of the elements at both pointers positions.

    Primary actions may be called to move pointers and swap elements at their positions.

    We call episode the sequence of actions applied to the environment and their corresponding states.
    The episode stops when the list is sorted.
    """

    def __init__(self, length=10, encoding_dim=32, hierarchy=True):

        assert dim > 0, "length must be a positive integer"
        self.dim = dim
        self.current_canvas = self._create_new_canvas()
        self.current_pixel_data = self.current_canvas.load()
        self.current_pix = np.array([dim/2]*2)
        self.encoding_dim = encoding_dim
        self.has_been_reset = False

        if hierarchy:
            self.programs_library = {'STOP': {'level': -1, 'recursive': False},
                                     'UP': {'level': 0, 'recursive': False},
                                     'DOWN': {'level': 0, 'recursive': False},
                                     'LEFT': {'level': 0, 'recursive': False},
                                     'RIGHT': {'level': 0, 'recursive': False},
                                     'UR_L': {'level': 1, 'recursive': False},
                                     'RU_L': {'level': 1, 'recursive': False},
                                     'SQUARE': {'level': 2, 'recursive': False}}
            for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
                self.programs_library[key]['index'] = idx

            self.prog_to_func = {'STOP': self._stop,
                                 'UP': self._up,
                                 'DOWN': self._down,
                                 'LEFT': self._left,
                                 'RIGHT': self._right}

            self.prog_to_precondition = {'STOP': self._stop_precondition,
                                         'UP': self._up_precondition,
                                         'DOWN': self._down_precondition,
                                         'LEFT': self._left_precondition,
                                         'RIGHT': self._right_precondition,
                                         'UR_L': self._ur_l_precondition,
                                         'RU_L': self._ru_l_precondition,
                                         'SQUARE': self._square_precondition,
                                        }

            self.prog_to_postcondition = {'UR_L': self._ur_l_postcondition,
                                          'RU_L': self._ru_l_postcondition,
                                          'SQUARE': self._square_postcondition,
                                         }

        else:
            # In no hierarchy mode, the only non-zero program is Bubblesort

            self.programs_library = {'STOP': {'level': -1, 'recursive': False},
                                     'UP': {'level': 0, 'recursive': False},
                                     'DOWN': {'level': 0, 'recursive': False},
                                     'LEFT': {'level': 0, 'recursive': False},
                                     'RIGHT': {'level': 0, 'recursive': False},
                                     'SQUARE': {'level': 1, 'recursive': False}}
            for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
                self.programs_library[key]['index'] = idx

            self.prog_to_func = {'STOP': self._stop,
                                 'UP': self._up,
                                 'DOWN': self._down,
                                 'LEFT': self._left,
                                 'RIGHT': self._right}

            self.prog_to_precondition = {'STOP': self._stop_precondition,
                                         'UP': self._up_precondition,
                                         'DOWN': self._down_precondition,
                                         'LEFT': self._left_precondition,
                                         'RIGHT': self._right_precondition,
                                         'SQUARE': self._square_precondition}

            self.prog_to_postcondition = {'SQUARE': self._square_postcondition}

            self.prog_to_postcondition = {'BUBBLESORT': self._bubblesort_postcondition}

        super(SimpleDrawEnv, self).__init__(self.programs_library, self.prog_to_func,
                                               self.prog_to_precondition, self.prog_to_postcondition)

    def _stop(self):
        """Do nothing. The stop action does not modify the environment."""
        pass

    def _stop_precondition(self):
        return True

    def _up(self):
        self.current_pix += [0,-1]

    def _up_precondition(self):
        self.current_pix[1] > 0

    def _down(self):
        self.current_pix += [0,1]

    def _down_precondition(self):
        self.current_pix[1] < self.dim-1

    def _left(self):
        self.current_pix += [-1,0]

    def _left_precondition(self):
        self.current_pix[0] > 0

    def _right(self):
        self.current_pix += [1,0]

    def _right_precondition(self):
        self.current_pix[0] < self.dim-1

    def _ur_l_precondition(self):
        return self.p1_pos > 0 or self.p2_pos > 0

    def _rshift_precondition(self):
        return self.p1_pos < self.length-1 or self.p2_pos < self.length-1

    def _bubble_precondition(self):
        bool = self.p1_pos == 0
        bool &= ((self.p2_pos == 0) or (self.p2_pos == 1))
        return bool

    def _reset_precondition(self):
        bool = True
        return bool

    def _bubblesort_precondition(self):
        bool = self.p1_pos == 0
        bool &= self.p2_pos == 0
        return bool

    def _compswap_postcondition(self, init_state, state):
        new_scratchpad_ints, new_p1_pos, new_p2_pos = init_state
        new_scratchpad_ints = np.copy(new_scratchpad_ints)
        if new_p1_pos == new_p2_pos and new_p2_pos < self.length-1:
            new_p2_pos += 1
        idx_left = min(new_p1_pos, new_p2_pos)
        idx_right = max(new_p1_pos, new_p2_pos)
        if new_scratchpad_ints[idx_left] > new_scratchpad_ints[idx_right]:
            new_scratchpad_ints[[idx_left, idx_right]] = new_scratchpad_ints[[idx_right, idx_left]]
        new_state = (new_scratchpad_ints, new_p1_pos, new_p2_pos)
        return self.compare_state(state, new_state)

    def _lshift_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos = init_state
        scratchpad_ints, p1_pos, p2_pos = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        if init_p1_pos > 0:
            bool &= p1_pos == (init_p1_pos-1)
        else:
            bool &= p1_pos == init_p1_pos
        if init_p2_pos > 0:
            bool &= p2_pos == (init_p2_pos-1)
        else:
            bool &= p2_pos == init_p2_pos
        return bool

    def _rshift_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos = init_state
        scratchpad_ints, p1_pos, p2_pos = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        if init_p1_pos < self.length-1:
            bool &= p1_pos == (init_p1_pos+1)
        else:
            bool &= p1_pos == init_p1_pos
        if init_p2_pos < self.length-1:
            bool &= p2_pos == (init_p2_pos+1)
        else:
            bool &= p2_pos == init_p2_pos
        return bool

    def _reset_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos = init_state
        scratchpad_ints, p1_pos, p2_pos = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        bool &= (p1_pos == 0 and p2_pos == 0)
        return bool

    def _bubblesort_postcondition(self, init_state, state):
        scratchpad_ints, p1_pos, p2_pos = state
        # check if list is sorted
        return np.all(scratchpad_ints[:self.length-1] <= scratchpad_ints[1:self.length])

    def _bubble_postcondition(self, init_state, state):
        new_scratchpad_ints, new_p1_pos, new_p2_pos = init_state
        new_scratchpad_ints = np.copy(new_scratchpad_ints)
        for idx in range(0, self.length-1):
            if new_scratchpad_ints[idx+1] < new_scratchpad_ints[idx]:
                new_scratchpad_ints[[idx, idx+1]] = new_scratchpad_ints[[idx+1, idx]]
        # bubble is expected to terminate with both pointers at the extreme left of the list
        new_p1_pos = self.length-1
        new_p2_pos = self.length-1
        new_state = (new_scratchpad_ints, new_p1_pos, new_p2_pos)
        return self.compare_state(state, new_state)

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

    def reset_env(self):
        """Reset the environment. The list are values are draw randomly. The pointers are initialized at position 0
        (at left position of the list).

        """
        self.scratchpad_ints = np.random.randint(10, size=self.length)
        current_task_name = self.get_program_from_index(self.current_task_index)
        if current_task_name == 'BUBBLE' or current_task_name == 'BUBBLESORT':
            init_pointers_pos1 = 0
            init_pointers_pos2 = 0
        elif current_task_name == 'RESET':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == 0 and init_pointers_pos2 == 0):
                    break
        elif current_task_name == 'LSHIFT':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == 0 and init_pointers_pos2 == 0):
                    break
        elif current_task_name == 'RSHIFT':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == self.length - 1 and init_pointers_pos2 == self.length - 1):
                    break
        elif current_task_name == 'COMPSWAP':
            init_pointers_pos1 = int(np.random.randint(0, self.length - 1))
            init_pointers_pos2 = int(np.random.choice([init_pointers_pos1, init_pointers_pos1 + 1]))
        else:
            raise NotImplementedError('Unable to reset env for this program...')

        self.p1_pos = init_pointers_pos1
        self.p2_pos = init_pointers_pos2
        self.has_been_reset = True

    def get_state(self):
        """Returns the current state.

        Returns:
            the environment state

        """
        assert self.has_been_reset, 'Need to reset the environment before getting states'
        return np.copy(self.scratchpad_ints), self.p1_pos, self.p2_pos

    def get_observation(self):
        """Returns an observation of the current state.

        Returns:
            an observation of the current state
        """
        assert self.has_been_reset, 'Need to reset the environment before getting observations'

        p1_val = self.scratchpad_ints[self.p1_pos]
        p2_val = self.scratchpad_ints[self.p2_pos]
        is_sorted = int(self._is_sorted())
        pointers_same_pos = int(self.p1_pos == self.p2_pos)
        pt_1_left = int(self.p1_pos == 0)
        pt_2_left = int(self.p2_pos == 0)
        pt_1_right = int(self.p1_pos == (self.length - 1))
        pt_2_right = int(self.p2_pos == (self.length - 1))
        p1p2 = np.eye(10)[[p1_val, p2_val]].reshape(-1)
        bools = np.array([
            pt_1_left,
            pt_1_right,
            pt_2_left,
            pt_2_right,
            pointers_same_pos,
            is_sorted
        ])
        return np.concatenate((p1p2, bools), axis=0)

    def get_observation_dim(self):
        """

        Returns:
            the size of the observation tensor
        """
        return 2 * 10 + 6

    def reset_to_state(self, state):
        """

        Args:
          state: a given state of the environment
        reset the environment is the given state

        """
        self.scratchpad_ints = state[0].copy()
        self.p1_pos = state[1]
        self.p2_pos = state[2]

    def _is_sorted(self):
        """Assert is the list is sorted or not.

        Args:

        Returns:
            True if the list is sorted, False otherwise

        """
        arr = self.scratchpad_ints
        return np.all(arr[:-1] <= arr[1:])

    def get_state_str(self, state):
        """Print a graphical representation of the environment state"""
        scratchpad = state[0].copy()  # check
        p1_pos = state[1]
        p2_pos = state[2]
        str = 'list: {}, p1 : {}, p2 : {}'.format(scratchpad, p1_pos, p2_pos)
        return str

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
        bool &= np.array_equal(state1[0], state2[0])
        bool &= (state1[1] == state2[1])
        bool &= (state1[2] == state2[2])
        return bool
