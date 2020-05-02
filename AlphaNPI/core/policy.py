import torch
from torch.nn import Linear, LSTMCell, Module, Embedding
from torch.nn.init import uniform_
from torch.distributions.beta import Beta
import torch.nn.functional as F
import numpy as np


device = 'cpu'


beta_pos = 0.0
beta_neg = 0.0

class CriticNet(Module):
    def __init__(self, hidden_size):
        super(CriticNet, self).__init__()
        self.l1 = Linear(hidden_size, hidden_size // 2)
        self.l2 = Linear(hidden_size // 2, 1)

    def forward(self, hidden_state):
        x = F.relu(self.l1(hidden_state))
        x = torch.tanh(self.l2(x))*100.0
        return x


class ActorNet(Module):
    def __init__(self, hidden_size, num_programs):
        super(ActorNet, self).__init__()
        self.program1 = Linear(hidden_size, hidden_size // 2)
        self.program2 = Linear(hidden_size // 2, num_programs)

    def forward(self, hidden_state):
        program = F.relu(self.program1(hidden_state))
        program = F.softmax(self.program2(program), dim=-1)
        return program


class ContinuousNet(Module):
    def __init__(self, hidden_size):
        super(ContinuousNet, self).__init__()
        self.beta1 = Linear(hidden_size, hidden_size)
        self.beta2 = Linear(hidden_size, hidden_size)
        self.beta3 = Linear(hidden_size, 2)

    def forward(self, hidden_state):
        beta = F.elu(self.beta1(hidden_state))
        beta = F.elu(self.beta2(beta))
        beta = F.softplus(self.beta3(beta))
        return beta


class Policy(Module):
    """This class represents the NPI policy containing the environment encoder, the key-value and program embedding
    matrices, the NPI core lstm and the value networks for each task.

    Args:
        encoder (:obj:`{HanoiEnvEncoder, ListEnvEncoder, RecursiveListEnvEncoder, PyramidsEnvEncoder}`):
        hidden_size (int): Dimensionality of the LSTM hidden state
        num_programs (int): Overall number of programs and size actor's output softmax vector
        num_non_primary_programs (int): Number of non-zero level programs, also number of rows in embedding matrix
        embedding_dim (int): Dimensionality of the programs' embedding vectors
        encoding_dim (int): Dimensionality of the environment observation's encoding
        indices_non_primary_programs (list): Non zero level programs' indices
        learning_rate (float, optional): Defaults to 10^-3.
    """

    def __init__(self, encoder, hidden_size, num_programs, num_non_primary_programs, embedding_dim,
                 encoding_dim, indices_non_primary_programs, learning_rate=1e-3, temperature=0.1):

        super(Policy, self).__init__()

        self._uniform_init = (-0.1, 0.1)

        self._hidden_size = hidden_size
        self.num_programs = num_programs
        self.num_non_primary_programs = num_non_primary_programs

        self.embedding_dim = embedding_dim
        self.encoding_dim = encoding_dim

        # Initialize networks
        self.Mprog = Embedding(num_non_primary_programs, embedding_dim)
        self.encoder = encoder

        self.lstm = LSTMCell(self.encoding_dim + self.embedding_dim, self._hidden_size)
        self.critic = CriticNet(self._hidden_size)
        self.actor = ActorNet(self._hidden_size, self.num_programs)
        self.beta_net = ContinuousNet(self._hidden_size)
        self.temperature = temperature
        self.min_val = 3.0
        self.val_count = 0
        self.entropy_lambda = 0.1
        self.init_networks()
        self.init_optimizer(lr=learning_rate)
        self.init_optimizer_beta(lr=1e-3)
        # Compute relative indices of non primary programs (to deal with task indices)
        self.relative_indices = dict(
            (prog_idx, relat_idx) for relat_idx, prog_idx in enumerate(indices_non_primary_programs))

    def init_networks(self):

        for p in self.encoder.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.lstm.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.critic.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.actor.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

    def init_optimizer(self, lr):
        '''Initialize the optimizer.

        Args:
            lr (float): learning rate
        '''
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def init_optimizer_beta(self, lr):
        '''Initialize the optimizer.

        Args:
            lr (float): learning rate
        '''

        for p in self.beta_net.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        self.optimizer_beta = torch.optim.Adam(self.beta_net.parameters(), lr=lr)

    def _one_hot_encode(self, digits, basis=6):
        """One hot encode a digit with basis. The digit may be None,
        the encoding associated to None is a vector full of zeros.

        Args:
          digits: batch (list) of digits
          basis:  (Default value = 6)

        Returns:
          a numpy array representing the 10-hot-encoding of the digit

        """
        encoding = torch.zeros(len(digits), basis)
        digits_filtered = list(filter(lambda x: x is not None, digits))

        if len(digits_filtered) != 0:
            tmp = [[idx for idx, digit in enumerate(digits) if digit is not None], digits_filtered]
            encoding[tmp] = 1.0
        return encoding

    def predict_on_batch(self, e_t, i_t, h_t, c_t):
        """Run one NPI inference.

        Args:
          e_t: batch of environment observation
          i_t: batch of calling program
          h_t: batch of lstm hidden state
          c_t: batch of lstm cell state

        Returns:
          probabilities over programs, value, new hidden state, new cell state

        """

        batch_size = len(i_t)
        s_t = self.encoder(e_t.view(batch_size, -1))
        relative_prog_indices = [self.relative_indices[idx] for idx in i_t]
        p_t = self.Mprog(torch.LongTensor(relative_prog_indices)).view(batch_size, -1)
        new_h, new_c = self.lstm(torch.cat([torch.flatten(s_t).view(batch_size, -1), p_t], -1),
                                 (h_t.view(batch_size, -1), c_t.view(batch_size, -1)))
        actor_out = self.actor(new_h)
        critic_out = self.critic(new_h)
        return actor_out, critic_out, new_h, new_c
        # return beta_out,actor_out, critic_out, new_h, new_c

    def predict_on_batch_beta(self, e_t, i_t, h_t, c_t):
        """Run one NPI inference.

        Args:
          e_t: batch of environment observation
          i_t: batch of calling program
          h_t: batch of lstm hidden state
          c_t: batch of lstm cell state

        Returns:
          probabilities over programs, value, new hidden state, new cell state

        """

        batch_size = len(i_t)
        s_t = self.encoder(e_t.view(batch_size, -1))
        relative_prog_indices = [self.relative_indices[idx] for idx in i_t]
        p_t = self.Mprog(torch.LongTensor(relative_prog_indices)).view(batch_size, -1)
        new_h, new_c = self.lstm(torch.cat([torch.flatten(s_t).view(batch_size, -1), p_t], -1),
                                 (h_t.view(batch_size, -1), c_t.view(batch_size, -1)))
        beta_out = self.beta_net(new_h)
        return beta_out


    def train_on_batch(self, batch):
        """perform optimization step.

        Args:
          batch (tuple): tuple of batches of environment observations, calling programs, lstm's hidden and cell states

        Returns:
          policy loss, value loss, total loss combining policy and value losses
        """
        # print("train on batch")
        e_t = torch.FloatTensor(np.stack(batch[0]))
        i_t = batch[1]
        batch_size = len(i_t)
        lstm_states = batch[2]
        h_t, c_t = zip(*lstm_states)
        h_t, c_t = torch.squeeze(torch.stack(list(h_t))), torch.squeeze(torch.stack(list(c_t)))
        policy_labels = torch.zeros(batch_size, self.num_programs)
        for i in range(batch_size):
            batch_len = batch[3][i].size()[1]
            policy_labels[i, 1:self.num_programs] = batch[3][i][0, batch_len - self.num_programs + 1:batch_len]
            policy_labels[i, 0] = torch.sum(batch[3][i][0:batch_len - self.num_programs + 1])

        value_labels = torch.stack(batch[4]).view(-1, 1)

        self.optimizer.zero_grad()

        policy_predictions, value_predictions, _, _ = self.predict_on_batch(e_t, i_t, h_t, c_t)
        policy_loss = -torch.mean(policy_labels * torch.log(policy_predictions), dim=-1).mean()
        value_loss = (torch.pow(value_predictions - value_labels, 2).mean())/5.0
        total_loss = (policy_loss + value_loss)
        total_loss.backward()
        self.optimizer.step()

        beta_l = batch[5]
        global weighted
        global w_count
        beta_probs = []
        for i in range(batch_size):
            batch_len = batch[3][i].size()[1] - self.num_programs + 1
            beta_probs.append( batch[3][i][0, 0:batch_len])
            beta_probs[i] = beta_probs[i] / beta_probs[i].sum()
            # print(value_labels[i])
            # print(beta_probs[i])
            # print(batch[5][i])
            # print(batch[3][i])
            # print()


        #TODO IF THIS WORKS CHANGE IT TO NOT UPDATE ANY PROGRAMS OTHER THAN MOVE BY LOOKING AT INDEX AFTER MOVE NODES
        # global beta_pos
        # global beta_neg
        loss_fn = torch.nn.KLDivLoss()
        for i in range(batch_size):
            # print(torch.max(batch[3][i],1))
            prog_position = torch.max(batch[3][i],1)[1].item()
            mcts_size = batch[3][i].size()[1]

            # print(mcts_size)
            # print(prog_position)
            # if prog_position != mcts_size-1 and prog_position != mcts_size-2:
            if prog_position < mcts_size - self.num_programs + 1:
                # print("beta training")
                print(value_labels[i])
                print(beta_probs[i])
                print(batch[5][i])
                # if value_labels[i] >0.0:
                #     beta_pos += 1.0
                # else:
                #     beta_neg += 1.0
                # if int( beta_pos + beta_neg)%100:
                #     print("beta ratio: " + str(beta_pos/(beta_neg+1)))
                self.optimizer_beta.zero_grad()
                beta_prediction = self.predict_on_batch_beta(e_t[i], [i_t[i]], h_t[i], c_t[i])
                # print(beta_prediction)
                dist = Beta(beta_prediction[0,0], beta_prediction[0,1])
                pdf_t = torch.exp(dist.log_prob(beta_l[i]))
                pdf_t = pdf_t/torch.sum(pdf_t)
                # print(pdf_t)
                pdf_t = torch.log(pdf_t)
                betaLoss = loss_fn(pdf_t, torch.unsqueeze(beta_probs[i], 0)) - self.entropy_lambda * dist.entropy()
                # print(betaLoss)
                # print(self.entropy_lambda*dist.entropy())
                # print()
                betaLoss.backward()
                self.optimizer_beta.step()
        return policy_loss, value_loss, total_loss


    def forward_once(self, e_t, i_t, h, c):
        """Run one NPI inference using predict.

        Args:
          e_t: current environment observation
          i_t: current program calling
          h: previous lstm hidden state
          c: previous lstm cell state

        Returns:
          probabilities over programs, value, new hidden state, new cell state, a program index sampled according to
          the probabilities over programs)

        """
        e_t = torch.FloatTensor(e_t)
        e_t, h, c = e_t.view(1, -1), h.view(1, -1), c.view(1, -1)
        with torch.no_grad():
            e_t = e_t.to(device)
            actor_out, critic_out, new_h, new_c = self.predict_on_batch(e_t, [i_t], h, c)
            beta_out = self.predict_on_batch_beta(e_t, [i_t], h, c)
            # beta_out, actor_out, critic_out, new_h, new_c = self.predict_on_batch(e_t, [i_t], h, c)
        return beta_out, actor_out, critic_out, new_h, new_c

    def init_tensors(self):
        """Creates tensors representing the internal states of the lstm filled with zeros.

        Returns:
            instantiated hidden and cell states
        """
        h = torch.zeros(1, self._hidden_size)
        c = torch.zeros(1, self._hidden_size)
        h, c = h.to(device), c.to(device)
        return h, c