# Contains NPI hyperparameters

# Environment agnostic space dimensions
encoding_dim = 128                           # dimension D of s_t
program_embedding_dim = 256                 # size P of program embedding vector

# LSTM hyper-param
hidden_size = 128                           # size of hidden state h_t

# Optimizer hyper-param
learning_rate = 1e-3                        # learning rate for the policy optimizer

# Curriculum hyper-params
reward_threshold = 14.00                    # reward threshold to increase the tasks levels in curriculum strategy

# MCTS hyper-params
number_of_simulations = 1000                 # number of simulations played before taking an action
c_puct = 0.65                             # trade-off exploration/exploitation in mcts
temperature = 0.5                          # coefficient to artificially increase variance in mcts policy distributions
level_closeness_coeff = 0.0001                 # importance given to higher level programs

# Training hyper-params
num_iterations = 50                        # total number of iterations, one iteration corresponds to one task
num_episodes_per_task = 10                  # number of episodes played for each new task attempted
batch_size = 16                           # training batch size
# batch_size = 256                            # training batch size
buffer_max_length = 4000                    # replay buffer max length
num_updates_per_episode = 2                 # number of gradient descents for every episode played
gamma = 0.97                                # discount factor to penalize long execution traces
proba_replay_buffer = 0.5                   # probability of sampling positive reward experience in buffer

# Validation hyper-params
num_validation_episodes = 10                # number of episodes played for validation
number_of_simulations_for_validation = 5    # number of simulations used in the tree for validation (when exploit = True)
