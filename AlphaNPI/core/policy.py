import torch
from torch.nn import Linear, LSTMCell, Module, Embedding
from torch.nn.init import uniform_
from torch.distributions.beta import Beta
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import scipy
from scipy import stats

device = 'cpu'


class CriticNet(Module):
    def __init__(self, hidden_size):
        super(CriticNet, self).__init__()
        self.l1 = Linear(hidden_size, hidden_size//2)
        self.l2 = Linear(hidden_size//2, 1)

    def forward(self, hidden_state):
        x = F.relu(self.l1(hidden_state))
        x = torch.sigmoid(self.l2(x))
        return x


class ActorNet(Module):
    def __init__(self, hidden_size, num_programs):
        super(ActorNet, self).__init__()
        self.program1 = Linear(hidden_size, hidden_size//2)
        self.program2 = Linear(hidden_size//2, num_programs)


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
        beta = F.softplus(self.beta1(hidden_state))
        beta = F.softplus(self.beta2(beta))
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

        self.entropy_lambda = 0.2
        self.init_networks()
        self.init_optimizer(lr=learning_rate)
        # self.init_optimizer_beta(lr=learning_rate)
        # Compute relative indices of non primary programs (to deal with task indices)
        self.relative_indices = dict((prog_idx, relat_idx) for relat_idx, prog_idx in enumerate(indices_non_primary_programs))

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
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)


    def init_optimizer_beta(self, lr):
        '''Initialize the optimizer.

        Args:
            lr (float): learning rate
        '''

        for p in self.beta_net.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        self.optimizer_beta = torch.optim.Adam(self.beta_net.parameters(), lr=lr*2)

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
        # print(relative_prog_indices)
        # print(relative_prog_indices)
        p_t = self.Mprog(torch.LongTensor(relative_prog_indices)).view(batch_size, -1)
        # print(p_t)
        new_h, new_c = self.lstm(torch.cat([torch.flatten(s_t).view(batch_size,-1), p_t], -1), (h_t.view(batch_size, -1), c_t.view(batch_size, -1)))
        # print(torch.flatten(s_t).view(batch_size,-1))
        # print(p_t)
        # print((h_t.view(batch_size, -1))
        # print(c_t.view(batch_size, -1))
        # print(new_h)
        # print(new_c)
        beta_out = self.beta_net(new_h)
        actor_out = self.actor(new_h)
        critic_out = self.critic(new_h)
        # print(actor_out[0])
        return actor_out, critic_out, beta_out, new_h, new_c
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
        new_h, new_c = self.lstm(torch.cat([torch.flatten(s_t).view(batch_size,-1), p_t], -1), (h_t.view(batch_size, -1), c_t.view(batch_size, -1)))
        beta_out = self.beta_net(new_h)
        return beta_out


    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.FloatTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)

    def train_on_batch(self, batch):
        """perform optimization step.

        Args:
          batch (tuple): tuple of batches of environment observations, calling programs, lstm's hidden and cell states

        Returns:
          policy loss, value loss, total loss combining policy and value losses
        """
        e_t = torch.FloatTensor(np.stack(batch[0]))
        i_t = batch[1]
        batch_size = len(i_t)
        lstm_states = batch[2]
        h_t, c_t = zip(*lstm_states)
        h_t, c_t = torch.squeeze(torch.stack(list(h_t))), torch.squeeze(torch.stack(list(c_t)))
        policy_labels = torch.zeros(batch_size,self.num_programs)
        for i in range(batch_size):
            batch_len = batch[3][i].size()[1]

            policy_labels[i,1:self.num_programs] = batch[3][i][0,batch_len-self.num_programs+1 :batch_len]

            policy_labels[i, 0] = torch.sum(batch[3][i][0:batch_len-self.num_programs+1])
        value_labels = torch.stack(batch[4]).view(-1, 1)
        beta_labels = batch[5]
        beta_probs = []
        beta_counts = []
        for i in range(batch_size):
            batch_len = batch[3][i].size()[1]
            betaL = batch[3][i][0,0:batch_len-self.num_programs+1]
            beta_counts.append(betaL.view(1,-1))
            betaL = torch.exp(betaL)/torch.exp(betaL).sum()
            betaL = betaL.view(1,-1)
            beta_probs.append(betaL)
        # print(batch[3][0][0])
        # print(beta_probs[0])
        # print()
        self.optimizer.zero_grad()

        policy_predictions, value_predictions, beta_predictions, _, _ = self.predict_on_batch(e_t, i_t, h_t, c_t)
        # beta_prediction, policy_predictions, value_predictions, _, _ = self.predict_on_batch(e_t, i_t, h_t, c_t)

        # print(torch.mean(beta_prediction,0))

        # total_betas  = 0
        # betaLoss = 0.0
        # for i in range(batch_size):
        #     dist = Beta(beta_prediction[i,0],beta_prediction[i,1])
        #     pdf_t = dist.log_prob(beta_labels[i])
        #     total_betas += pdf_t.size()[1]
        #     temp = torch.log(beta_probs[i])
        #     betaLoss += (pdf_t -temp).sum() - self.entropy_lambda * dist.entropy()
        #     # print(dist.entropy())
        #
        # betaLoss /= float(total_betas)
        # betaLoss /= 3.0

        policy_loss = -torch.mean(policy_labels * torch.log(policy_predictions), dim=-1).mean()

        value_loss = torch.pow(value_predictions - value_labels, 2).mean()

        # total_loss = (policy_loss + value_loss + betaLoss)

        # total_loss = (policy_loss + value_loss)
        # # print("betaLoss: " + str(betaLoss.item()) + "    totalLoss: " + str(total_loss.item()) + "    policyLoss: " + str(policy_loss.item()) + "    valueLoss: " + str(value_loss.item()))

        # # a = list(self.beta_net.parameters())[0].clone()
        # total_loss.backward()
        # self.optimizer.step()
        # b = list(self.beta_net.parameters())[0].clone()
        # print(torch.equal(a.data, b.data))


        # beta_param_labels = []
        # for i in range(len(beta_labels)):
        #     # print(beta_counts[i])
        #     beta_label_distr = beta_labels[i].repeat_interleave((1000*beta_counts[i]).squeeze().type(torch.LongTensor), dim=1)
        #     beta_label_distr = beta_label_distr.squeeze().data.numpy()

        #     a, b, _, _ = stats.beta.fit(beta_label_distr)
        #     beta_param_labels.append(torch.tensor([[a, b]]))

        # beta_param_labels = torch.cat(beta_param_labels, dim=0)

        # beta_prediction = self.predict_on_batch_beta(e_t, i_t, h_t, c_t)
        # beta_pred_dist = Beta(beta_prediction[:, 0], beta_prediction[:, 1])
        # beta_target_dist = Beta(beta_param_labels[:, 0], beta_param_labels[:, 1])
        # betaLoss = torch.distributions.kl.kl_divergence(beta_pred_dist, beta_target_dist).mean()





        # self.optimizer_beta.zero_grad()
        # beta_prediction = self.predict_on_batch_beta(e_t, i_t, h_t, c_t)

        beta_dim = torch.tensor([b.squeeze().size()[0] for b in beta_labels])
        beta_prediction_repeated = beta_predictions.repeat_interleave(beta_dim, dim=0)

        dist = Beta(beta_prediction_repeated[:, 0], beta_prediction_repeated[:, 1])
        
        pdf_t = dist.log_prob(torch.cat(beta_labels, dim=1).flatten())
        temp = torch.log(torch.cat(beta_probs, dim=1).flatten())

        # pdf_t[abs(pdf_t) == float('inf')] = 0
        # temp[abs(temp) == float('inf')] = 0

        # print((pdf_t - temp).pow(2))
        # print(dist.entropy())

        betaLoss = ((pdf_t - temp)*pdf_t + self.entropy_lambda * dist.entropy()).mean()
        

        # for i in range(batch_size):
        #     dist = Beta(beta_prediction[i, 0], beta_prediction[i, 1])
        #     pdf_t = dist.log_prob(beta_labels[i])
        #     # print(beta_prediction[i].tolist())
        #     # print(beta_labels[i].mean())
        #     total_betas += pdf_t.size()[1]
        #     temp = torch.log(beta_probs[i])
        #     betaLoss += (pdf_t - temp).pow(2).sum() - self.entropy_lambda * dist.entropy()

        #     sum_pdf += pdf_t.sum()
        #     sum_beta_probs += temp.sum()
        #     entropy += dist.entropy()
        #     # print(dist.entropy())

        #     print(f"Beta VC: {beta_labels[i]} \nProb: {beta_probs[i]}")

        # print(f"sum_pdf: {sum_pdf}    sum_beta_probs: {sum_beta_probs}    entropy: {entropy}")


        # betaLoss /= float(total_betas)
        # c = list(self.beta_net.parameters())[0].clone()

        total_loss = (policy_loss + value_loss + betaLoss)
        total_loss.backward()
        self.optimizer.step()

        print("betaLoss: " + str(betaLoss.item()) + "    totalLoss: " + str(total_loss.item()) + "    policyLoss: " + str(policy_loss.item()) + "    valueLoss: " + str(value_loss.item()))

        # betaLoss.backward()
        # self.optimizer_beta.step()
        # d = list(self.beta_net.parameters())[0].clone()
        # print(torch.equal(c.data, d.data))


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
            actor_out, critic_out, beta_out, new_h, new_c = self.predict_on_batch(e_t, [i_t], h, c)
            # beta_out = self.predict_on_batch_beta(e_t, [i_t], h, c)
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

