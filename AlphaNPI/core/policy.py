import torch
from torch.nn import Linear, LSTMCell, Module, Embedding
from torch.nn.init import uniform_
from torch.distributions.beta import Beta
import torch.nn.functional as F
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
        self.beta1 = Linear(hidden_size, hidden_size//2)
        self.beta2 = Linear(hidden_size//2, 2)

    def forward(self, hidden_state):
        beta = F.relu(self.beta1(hidden_state))
        beta = F.softplus(self.beta2(beta))
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

        self.entropy_lambda = 0.1

        self.init_networks()
        self.init_optimizer(lr=learning_rate)

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

        for p in self.beta_net.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

    def init_optimizer(self, lr):
        '''Initialize the optimizer.

        Args:
            lr (float): learning rate
        '''
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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
        return beta_out,actor_out, critic_out, new_h, new_c

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
        for i in range(batch_size):
            batch_len = batch[3][i].size()[1]
            betaL = batch[3][i][0,0:batch_len-self.num_programs+1]
            betaL = betaL/betaL.sum()
            betaL = betaL.view(1,-1)
            beta_probs.append(betaL)


        self.optimizer.zero_grad()



        beta_prediction, policy_predictions, value_predictions, _, _ = self.predict_on_batch(e_t, i_t, h_t, c_t)

        # print(torch.mean(beta_prediction,0))

        betaLoss = 0.0
        total_betas = 0
        for i in range(batch_size):
            dist = Beta(beta_prediction[i,0],beta_prediction[i,1])

            pdf_t = torch.zeros(1,beta_probs[i].size()[1])
            for j in range(beta_probs[i].size()[1]):
                # print(str(i) + "   " + str(j) + "   "  + str(beta_labels[i][0,j]) +"    " +str(dist.log_prob(beta_labels[i][0,j])) + "   " +str(beta_prediction[i,0]) + "   " +str(beta_prediction[i,1]) )
                pdf_t[0, j] = dist.log_prob(beta_labels[i][0,j])
                total_betas +=1
            betaLoss += (pdf_t - torch.log(beta_probs[i])).sum() - self.entropy_lambda * dist.entropy()



        betaLoss =  betaLoss / float(total_betas)

        policy_loss = -torch.mean(policy_labels * torch.log(policy_predictions), dim=-1).mean()

        value_loss = torch.pow(value_predictions - value_labels, 2).mean()



        total_loss = (policy_loss + value_loss + betaLoss)
        # print("betaLoss: " + str(betaLoss.item()) + "    totalLoss: " + str(total_loss.item()) + "    policyLoss: " + str(policy_loss.item()) + "    valueLoss: " + str(value_loss.item()))
        total_loss.backward()
        self.optimizer.step()

        return policy_loss, value_loss, total_loss









        # i_t = batch[1]
        # batch_size = len(i_t)
        # e_t = torch.FloatTensor(np.stack(batch[0]))
        # e_t = e_t.view(len(i_t), -1)
        # lstm_states = batch[2]
        # h_t, c_t = zip(*lstm_states)
        # h_t, c_t = torch.squeeze(torch.stack(list(h_t))), torch.squeeze(torch.stack(list(c_t)))
        # num_policies = self.num_programs + self.max_wide - 1
        # expanded_policies = torch.zeros(batch_size, num_policies)
        # for i in range(batch_size):
        #     # print(torch.cat((torch.zeros(1,num_policies-batch[3][i].size()[1]),batch[3][i]),1).size())
        #     # print(torch.zeros(1,num_policies-batch[3][i].size()[1]))
        #     # print(batch[3][i])
        #     # print(torch.cat((torch.zeros(1,num_policies-batch[3][i].size()[1]),batch[3][i]),1))
        #     expanded_policies[i] = torch.cat((torch.zeros(1,num_policies-batch[3][i].size()[1]),batch[3][i]),1)
        # # print(expanded_policies[0])
        # # policy_labels = torch.squeeze(torch.stack(expanded_policies)).view(batch_size,-1 )
        # policy_labels = expanded_policies
        # # print(policy_labels.size())
        #
        # # for i in range(batch_size):
        # #     if policy_labels.size()[i].size()
        # value_labels = torch.stack(batch[4]).view(-1, 1)
        # # print(value_labels)
        # self.optimizer.zero_grad()
        # policy_predictions, value_predictions, _, _ = self.predict_on_batch(e_t, i_t, h_t, c_t)
        # # print(policy_predictions)
        # # print()
        # # policy_loss = -torch.mean(policy_labels * torch.log(policy_predictions), dim=-1).mean()
        #
        # # print(policy_predictions[0][0].size())
        # prob_action = np.zeros((batch_size,policy_predictions[0][0].size()[0]),dtype=float)
        # print(prob_action)
        # entropy_loss = np.zeros((batch_size,1),dtype=float)
        # for i in range(batch_size):
        #     beta = Beta(policy_predictions[1][i][0], policy_predictions[1][i][1])
        #     prob_action[i,0] = beta.log_prob(beta.sample())
        #     prob_action[i,1] = beta.log_prob(beta.sample())
        #     prob_action[i,2] = beta.log_prob(beta.sample())
        #     entropy_loss[i,0] = self.entropy_lambda * beta.entropy()
        # # print(prob_action)
        # prob_action = torch.from_numpy(prob_action)
        # entropy_loss = torch.from_numpy(entropy_loss)
        # log_mcts = self.temperature * torch.log(policy_labels)
        #
        # # print(prob_action.size())
        # # print(log_mcts.size())
        # # print(prob_action)
        # # print(log_mcts)
        # # print()
        # with torch.no_grad():
        #     modified_kl = prob_action - log_mcts
        # # print(modified_kl)
        # # print(prob_action)
        # # print()
        # policy_loss = -modified_kl * (torch.log(modified_kl) + prob_action)
        # # entropy_loss = self.entropy_lambda * beta.entropy()
        #
        # # print(policy_loss)
        # # print(entropy_loss)
        # # print()
        # policy_network_loss = policy_loss + entropy_loss
        # value_network_loss = torch.pow(value_predictions - value_labels, 2).mean()
        # # print(policy_network_loss)
        # # print(value_network_loss)
        # # print()
        #
        # total_loss = torch.sum(((policy_network_loss + value_network_loss) / 2),dtype=float)
        # # print(total_loss)
        # total_loss.backward()
        # self.optimizer.step()
        #
        # return policy_network_loss, value_network_loss, total_loss

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
            beta_out, actor_out, critic_out, new_h, new_c = self.predict_on_batch(e_t, [i_t], h, c)
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

