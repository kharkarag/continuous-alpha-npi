import numpy as np

class ContinuousReplayBuffer():
    '''
    This class represents a replay buffer memory in which traces generated by the MCTS are stored.
    '''

    def __init__(self, max_length, task_ids, p1=0.8):
        self.task_ids = task_ids
        self.memory_task = dict((task_id, []) for task_id in self.task_ids)
        self.stack = []
        self.max_length = max_length
        self.p1 = p1

    def get_memory_length(self):
        return len(self.stack)

    def append_trace(self, trace):
        '''
        Add a newly generated execution trace to the memory buffer.
        Args:
            trace: a sequence of [(e_0, i_0, (h_0, c_0), pi_0, r_0), ... , (e_T, i_T, (h_T, c_T), pi_T, r_T)]
        '''
        for tup in trace:
            if len(self.stack) >= self.max_length:
                t_id = self.stack[0][1]
                del self.memory_task[t_id][0]
                del self.stack[0]
            task_id = tup[1]
            self.memory_task[task_id].append(tup)
            self.stack.append(tup)

    def _sample_sub_batch(self, batch_size, memory):
        indices = np.arange(len(memory))
        sampled_indices = np.random.choice(indices, size=batch_size, replace=(batch_size > len(memory)))
        batch = [[], [], [], [], [],[]]
        for i in sampled_indices:
            for k in range(6):
                batch[k].append(memory[i][k])
        return batch

    def sample_batch(self, batch_size):
        '''
        Sample in the memory a batch of experience.
        Args:
            batch_size: the batch size
        Returns:
            a list [batch of e_t, batch of i_t, batch of (h_t, c_t), batch of pi_t, batch of r_t]
        '''
        memory = []
        for task_id in self.memory_task:
            if len(self.memory_task[task_id]) > 0:
                memory += self.memory_task[task_id]

        if len(memory) == 0:
            return None
        else:
            batch = self._sample_sub_batch(batch_size, memory)

        return batch if batch else None

    def empty_memory(self):
        '''
        Empty the replay memory.
        '''
        self.memory_task = dict((task_id, []) for task_id in self.task_ids)
        self.stack = []