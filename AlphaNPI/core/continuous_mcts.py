# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

import numpy as np
import torch
from scipy import stats
from torch.distributions.beta import Beta

class ContinuousMCTS:
    """This class is used to perform a search over the state space for different paths by building
    a tree of visited states. Then this tree is used to get an estimation distribution of
    utility over actions.

    Args:
      policy: Policy to be used as a prior over actions given an state.
      c_puct: Constant that modifies the exploration-exploitation tradeoff of the MCTS algorithm.
      env: The environment considered.
      task_index: The index of the task (index of the corresponding program) we are trying to solve.
      number_of_simulations: The number of nodes that we will be visiting when building an MCTS tree.
      temperature: Another parameter that balances exploration-exploitation in MCTS by adding noise to the priors output by the search.
      max_depth_dict: Dictionary that maps a program level to the allowed number of actions to execute a program of that level
      use_dirichlet_noise: Boolean authorizes or not addition of dirichlet noise to prior during simulations to encourage exploration
      dir_epsilon: Proportion of the original prior distribution kept in the newly-updated prior distribution with dirichlet noise
      dir_noise: Parameter of the Dirichlet distribution
      #BELOW LINE IS FROM ORIGINAL NPI AND IS INCORRECT.  EXPLOIT USES ARGMAX
      exploit: Boolean if True leads to sampling from the mcts visit policy instead of taking the argmax
      gamma: discount factor, reward discounting increases with depth of trace
      save_sub_trees: Boolean to save in a node the sub-execution trace of a non-zero program
      recursion_depth: Recursion level of the calling tree
      max_recursion_depth: Max recursion level allowed
      qvalue_temperature: Induces tradeoff between mean qvalue and max qvalue when estimating Q in PUCT criterion
      recursive_penalty: Penalty applied to discounted reward if recursive program does not call itself
    """

    def __init__(self, policy, env, task_index, level_closeness_coeff=1.0,
                 c_puct=1.0, number_of_simulations=100, max_depth_dict={1: 20, 2: 20, 3: 20},
                 temperature=1.0, use_dirichlet_noise=False,
                 dir_epsilon=0.25, dir_noise=0.03, exploit=False, gamma=0.97, save_sub_trees=False,
                 recursion_depth=0, max_recursion_depth=500, qvalue_temperature=1.0, recursive_penalty=0.9,cpw = 1, kappa = 0.5):

        self.policy = policy
        self.c_puct = c_puct
        self.level_closeness_coeff = level_closeness_coeff
        self.env = env
        self.task_index = task_index
        self.task_name = env.get_program_from_index(task_index)
        self.recursive_task = env.programs_library[self.task_name]['recursive']
        self.recursive_penalty = recursive_penalty
        self.number_of_simulations = number_of_simulations
        self.temperature = temperature
        self.max_depth_dict = max_depth_dict
        self.dirichlet_noise = use_dirichlet_noise
        self.dir_epsilon = dir_epsilon
        self.dir_noise = dir_noise
        self.exploit = exploit
        self.gamma = gamma
        self.save_sub_trees = save_sub_trees
        self.recursion_depth = recursion_depth
        self.max_recursion_depth = max_recursion_depth
        self.qvalue_temperature = qvalue_temperature
        self.cpw = cpw
        self.kappa = kappa
        self.max_wide = 40

        # record if all sub-programs executed correctly (useful only for programs of level > 1)
        self.clean_sub_executions = True

        # recursive trees parameters
        self.sub_tree_params = {'number_of_simulations': 5, 'max_depth_dict': self.max_depth_dict,
            'temperature': self.temperature, 'c_puct': self.c_puct, 'exploit': True,
            'level_closeness_coeff': self.level_closeness_coeff, 'gamma': self.gamma,
            'save_sub_trees': self.save_sub_trees, 'recursion_depth': recursion_depth+1}




    #Keep track of all nodes and the continuous actions they have available
    def continuous_children(self,program_index, betaD):
        #Go through each program and expand if needed
        mask = self.env.get_mask_over_actions(program_index)
        c_actions = {}
        #This will give the index for each available program
        for prog_index in [prog_idx for prog_idx, x in enumerate(mask) if x == 1]:
            pname = self.env.get_program_from_index(prog_index)
            if self.env.programs_library[pname]['continuous'] == True:
                crange  = self.env.programs_library[pname]['crange']
                dist = Beta(betaD[0], betaD[1])
                new_cval = crange[0] + crange[1] * dist.sample()

                # Dist_val = np.random.beta(betaD[0],betaD[1])
                # # print(Dist_val)
                # new_cval = crange[0] + crange[1] * Dist_val

                c_actions[prog_index] = {"cval": new_cval}
        return c_actions



    def check_widening(self, node):
        continuous_children_num = int(self.cpw * node["visit_count"] ** self.kappa)
        continuous_children_previous= int(self.cpw * (node["visit_count"]-1.0) ** self.kappa)
        #If m(s) =int( cpw * n(s)^(kappa) ) increased you need to add another node
        if continuous_children_num> continuous_children_previous:
            program_index, observation, env_state, h, c, depth, Beta_Parameters = (
                node["program_index"],
                node["observation"],
                node["env_state"],
                node["h_lstm"],
                node["c_lstm"],
                node["depth"],
                node["Beta_Parameters"]
            )

            mask = self.env.get_mask_over_actions(program_index)

            # This will give the index for each available program
            for prog_index in [prog_idx for prog_idx, x in enumerate(mask) if x == 1]:
                pname = self.env.get_program_from_index(prog_index)
                if self.env.programs_library[pname]['continuous'] == True:
                    child_prior= 0.0
                    for n in node["childs"]:
                        if self.env.get_program_from_index(n["program_from_parent_index"]) == pname:
                            child_prior = n["prior"]
                    crange = self.env.programs_library[pname]['crange']

                    dist = Beta(Beta_Parameters[0], Beta_Parameters[1])
                    new_cval = crange[0] + crange[1] * dist.sample()
                    # print(new_cval)
                    # Dist_val = np.random.beta(Beta_Parameters[0], Beta_Parameters[1])
                    # # print(Dist_val)
                    # new_cval = crange[0] + crange[1] * Dist_val

                    #Need to do a search to get the sum of the priors of the other programs of same type and mult by dist

                    new_child = {
                        "parent": node,
                        "childs": [],
                        "visit_count": 0.0,
                        "total_action_value": [],
                        "prior": child_prior ,
                        "program_from_parent_index": prog_index,
                        # This is making the same actions availible to child as parent
                        "program_index": program_index,
                        "observation": observation,
                        "env_state": env_state,
                        "h_lstm": h.clone(),
                        "c_lstm": c.clone(),
                        "selected": False,
                        "depth": depth + 1,
                        "cval": new_cval,
                        "Beta_Parameters":Beta_Parameters,
                        "Parent_program_name":self.env.get_program_from_index(prog_index)
                    }
                    node["childs"].insert(0,new_child)



    def _expand_node(self, node):
        """Used for previously unvisited nodes. It evaluates each of the possible child and
        initializes them with a score derived from the prior output by the policy network.

        Args:
          node: Node to be expanded

        Returns:
          node now expanded, value, hidden_state, cell_state

        """
        #This should be fine as it will already have the widened program options
        program_index, observation, env_state, h, c, depth = (
            node["program_index"],
            node["observation"],
            node["env_state"],
            node["h_lstm"],
            node["c_lstm"],
            node["depth"]
        )

        with torch.no_grad():
            mask = self.env.get_mask_over_actions(program_index)
            beta_out, priors, value, new_h, new_c = self.policy.forward_once(observation, program_index, h, c)
            # print(priors)
            # print(value)
            betaD = torch.flatten(beta_out)
            # print(str(betaD[0])+  '  ' + str(betaD[1]) + '  '  + str(np.random.beta(betaD[0],betaD[1])))
            # mask actions
            priors = priors * torch.FloatTensor(mask)
            priors = torch.squeeze(priors)
            priors = priors.cpu().numpy()
            if self.dirichlet_noise:
                priors = (1 - self.dir_epsilon) * priors + self.dir_epsilon * np.random.dirichlet([self.dir_noise] * priors.size)


        node["Beta_Parameters"] = betaD
        c_children = self.continuous_children(program_index, betaD)
        # Initialize its children with its probability of being chosen
        for prog_index in [prog_idx for prog_idx, x in enumerate(mask) if x == 1]:
            # program_name = self.env.get_program_from_index(prog_index)
            #May want to change this.  It relies on it being a new node so there will only be one continuous value as no widening will have happened
            cval = None

            if prog_index in c_children:
                cval = c_children[prog_index]["cval"]
                # print(cval)
            new_child = {
                "parent": node,
                "childs": [],
                "visit_count": 0.0,
                "total_action_value": [],
                "prior": float(priors[prog_index]),
                "program_from_parent_index": prog_index,
                #This is making the same actions availible to child as parent
                "program_index": program_index,
                "observation": observation,
                "env_state": env_state,
                "h_lstm": new_h.clone(),
                "c_lstm": new_c.clone(),
                "selected": False,
                "depth": depth + 1,
                "cval": cval,
                "Beta_Parameters": betaD,
                "Parent_program_name": self.env.get_program_from_index(prog_index)
            }
            node["childs"].append(new_child)


        # This reward will be propagated backwards through the tree
        value = float(value)
        return node, value, new_h.clone(), new_c.clone()


    def _compute_q_value(self, node):
        if node["visit_count"] > 0.0:
            values = torch.FloatTensor(node['total_action_value'])
            softmax = torch.exp(self.qvalue_temperature * values)
            softmax = softmax / softmax.sum()
            q_val_action = float(torch.dot(softmax, values))
        else:
            q_val_action = 0.0
        return q_val_action

    def _estimate_q_val(self, node):
        """Estimates the Q value over possible actions in a given node, and returns the action
        and the child that have the best estimated value.

        Args:
          node: Node to evaluate its possible actions.

        Returns:
          best child found from this node.

        """

        best_val = -np.inf
        best_child = None
        # Iterate all the children to fill up the node dict and estimate Q val.
        # Then track the best child found according to the Q value estimation
        # print(len(node["childs"]))
        for child in node["childs"]:
            if child["prior"] > 0.0:
                q_val_action = self._compute_q_value(child)

                action_utility = (self.c_puct * child["prior"] * np.sqrt(node["visit_count"])
                                  * (1.0 / (1.0 + child["visit_count"])))
                q_val_action += action_utility
                parent_prog_lvl = self.env.programs_library[self.env.idx_to_prog[node['program_index']]]['level']
                action_prog_lvl = self.env.programs_library[self.env.idx_to_prog[child['program_from_parent_index']]]['level']

                if parent_prog_lvl == action_prog_lvl:
                    # special treatment for calling the same program
                    action_level_closeness = self.level_closeness_coeff * np.exp(-1)
                elif action_prog_lvl > -1:
                    action_level_closeness = self.level_closeness_coeff * np.exp(-(parent_prog_lvl - action_prog_lvl))
                else:
                    # special treatment for STOP action
                    action_level_closeness = self.level_closeness_coeff * np.exp(-1)

                q_val_action += action_level_closeness
                if q_val_action > best_val:
                    best_val = q_val_action
                    best_child = child

        if best_child == None:
            print("None Child")

        return best_child


    def _sample_policy(self, root_node):
        """Sample an action from the policies and q_value distributions that were previously sampled.

        Args:
          root_node: Node to choose the best action from. It should be the root node of the tree.

        Returns:
          Tuple containing the sampled action and the probability distribution build normalizing visits_policy.
        """
        mask = self.env.get_mask_over_actions( root_node["program_index"])
        pad = mask.shape[0] -len(np.nonzero(mask)[0])
        visits_policy = []
        for i, child in enumerate(root_node["childs"]):
            if child["prior"] > 0.0:
                visits_policy.append([i, child["visit_count"]])

        mcts_policy = torch.zeros(1, len(root_node["childs"])+pad)
        for i, visit in visits_policy:
            mcts_policy[0, i] = visit

        if self.exploit:
            mcts_policy = mcts_policy / mcts_policy.sum()
            return mcts_policy, int(torch.argmax(mcts_policy))

        else:
            mcts_policy = torch.pow(mcts_policy, self.temperature)
            mcts_policy = mcts_policy / mcts_policy.sum()
            return mcts_policy, int(torch.multinomial(mcts_policy, 1)[0, 0])



    def _run_simulation(self, node):
        """Run one simulation in tree. This function is recursive.

        Args:
          node: root node to run the simulation from
          program_index: index of the current calling program

        Returns:
            (if the max depth has been reached or not, if a node has been expanded or not, node reached at the end of the simulation)

        """

        stop = False
        max_depth_reached = False
        max_recursion_reached = False
        has_expanded_a_node = False
        value = None
        program_level = self.env.get_program_level_from_index(node['program_index'])

        while not stop and not max_depth_reached and not has_expanded_a_node and self.clean_sub_executions and not max_recursion_reached:

            if node['depth'] >= self.max_depth_dict[program_level]:
                max_depth_reached = True

            elif len(node['childs']) == 0:
                _, value, state_h, state_c = self._expand_node(node)
                has_expanded_a_node = True

            else:
                node = self._estimate_q_val(node)
                program_to_call_index = node['program_from_parent_index']
                program_to_call = self.env.get_program_from_index(program_to_call_index)
                if program_to_call_index == self.env.programs_library['STOP']['index']:
                    stop = True

                elif self.env.programs_library[program_to_call]['level'] == 0:
                    observation = self.env.act(program_to_call, node["cval"])
                    node['observation'] = observation
                    node['env_state'] = self.env.get_state()

                else:
                    # check if call corresponds to a recursive call
                    if program_to_call_index == self.task_index:
                        self.recursive_call = True
                    # if never been done, compute new tree to execute program
                    if node['visit_count'] == 0.0:

                        if self.recursion_depth >= self.max_recursion_depth:
                            max_recursion_reached = True
                            continue

                        sub_mcts_init_state = self.env.get_state()
                        sub_mcts = ContinuousMCTS(self.policy, self.env, program_to_call_index, **self.sub_tree_params)
                        sub_trace = sub_mcts.sample_execution_trace()
                        sub_task_reward, sub_root_node = sub_trace[7], sub_trace[6]

                        # if save sub tree is true, then store sub root node
                        if self.save_sub_trees:
                            node['sub_root_node'] = sub_root_node
                        # allows tree saving of first non zero program encountered

                        # check that sub tree executed correctly
                        self.clean_sub_executions &= (sub_task_reward > -1.0)
                        if not self.clean_sub_executions:
                            print('program {} did not execute correctly'.format(program_to_call))
                            self.programs_failed_indices.append(program_to_call_index)
                            #self.programs_failed_indices += sub_mcts.programs_failed_indices
                            self.programs_failed_initstates.append(sub_mcts_init_state)

                        observation = self.env.get_observation()
                    else:
                        self.env.reset_to_state(node['env_state'])
                        observation = self.env.get_observation()

                    node['observation'] = observation
                    node['env_state'] = self.env.get_state()

        return max_depth_reached, has_expanded_a_node, node, value

    def _play_episode(self, root_node):
        """Performs an MCTS search using the policy network as a prior and returns a sequence of improved decisions.

        Args:
          root_node: Root node of the tree.

        Returns:
            (Final node reached at the end of the episode, boolean stating if the max depth allowed has been reached).

        """
        stop = False
        max_depth_reached = False

        while not stop and not max_depth_reached and self.clean_sub_executions:

            program_level = self.env.get_program_level_from_index(root_node['program_index'])
            # tag node as from the final execution trace (for visualization purpose)
            root_node["selected"] = True

            if root_node['depth'] >= self.max_depth_dict[program_level]:
                max_depth_reached = True

            else:
                env_state = root_node["env_state"]

                # record obs, progs and lstm states only if they correspond to the current task at hand
                self.lstm_states.append((root_node['h_lstm'], root_node['c_lstm']))
                self.programs_index.append(root_node['program_index'])
                self.observations.append(root_node['observation'])
                self.previous_actions.append(root_node['program_from_parent_index'])
                self.rewards.append(None)

                # Spend some time expanding the tree from your current root node
                for j in range(self.number_of_simulations):
                    # run a simulation
                    # print(root_node['depth'])
                    # print("play episode number: " +str(j))
                    self.recursive_call = False
                    simulation_max_depth_reached, has_expanded_node, node, value = self._run_simulation(root_node)

                    # get reward
                    if not simulation_max_depth_reached and not has_expanded_node:
                        # if node corresponds to end of an episode, backprogagate real reward
                        reward = self.env.get_reward() - root_node['depth']/1000.0
                        # print("reward: " + str(reward))
                        if reward > 0:
                            value = self.env.get_reward() * (self.gamma ** node['depth'])
                            # print("value: " + str(value))
                            if self.recursive_task and not self.recursive_call:
                                # if recursive task but do not called itself, add penalization
                                value -= self.recursive_penalty
                        else:
                            value = 0.0

                    elif simulation_max_depth_reached:
                        # if episode stops because the max depth allowed was reached, then reward = -1
                        value = 0.0

                    value = float(value)


                    #THIS IS THE ONLY PLACE VISIT COUNT IS INCREMENTED SO IT WIDENS HERE
                    # Propagate information backwards
                    while node["parent"] is not None:
                        node["visit_count"] += 1
                        node["total_action_value"].append(value)
                        self.check_widening(node)
                        node = node["parent"]
                    # Root node is not included in the while loop
                    self.root_node["total_action_value"].append(value)
                    self.root_node["visit_count"] += 1
                    self.check_widening(self.root_node)
                    # Go back to current env state
                    self.env.reset_to_state(env_state)

                # Sample next action
                # print(type(root_node))
                # print(type(self._sample_policy(root_node)))
                mcts_policy, program_to_call_index = self._sample_policy(root_node)
                num_continuous = len(root_node["childs"])-self.env.get_num_programs()+2
                pname = self.env.get_program_from_index(root_node["childs"][0]["program_from_parent_index"])
                crange = self.env.programs_library[pname]['crange']
                cont_vals =torch.zeros(1,num_continuous)
                for i in range(num_continuous):
                    cont_vals[0, i] = (root_node["childs"][i]["cval"]-crange[0])/crange[1]
                if self.env.get_program_from_index(root_node["childs"][program_to_call_index]["program_index"]) == self.env.get_program_from_index(self.task_index):
                    self.global_recursive_call = True



                # Set new root node

                root_node = root_node["childs"][program_to_call_index]
                # print(str(root_node["Parent_program_name"]) + "    " +str(root_node["cval"]) )
                # Record mcts policy
                self.mcts_policies.append(mcts_policy)
                self.cvals.append(cont_vals)
                # Apply chosen action
                if self.env.get_program_from_index(root_node["program_from_parent_index"]) == 'STOP':
                    stop = True
                else:
                    self.env.reset_to_state(root_node["env_state"])

        return root_node, max_depth_reached


    def sample_execution_trace(self):
        """
        Args:
          init_observation: initial observation before playing an episode

        Returns:
            (a sequence of (e_t, i_t), a sequence of probabilities over programs, a sequence of (h_t, c_t), if the maximum depth allowed has been reached)
        """

        # start the task
        init_observation = self.env.start_task(self.task_index)
        with torch.no_grad():
            state_h, state_c = self.policy.init_tensors()
            self.env_init_state = self.env.get_state()

            self.root_node = {
                "parent": None,
                "childs": [],
                "visit_count": 1,
                "total_action_value": [],
                "prior": None,
                "program_index": self.task_index,
                "program_from_parent_index": None,
                "observation": init_observation,
                "env_state": self.env_init_state,
                "h_lstm": state_h.clone(),
                "c_lstm": state_c.clone(),
                "depth": 0,
                "selected": True
            }

            # prepare empty lists to store trajectory
            self.programs_index = []
            self.observations = []
            self.previous_actions = []
            self.mcts_policies = []
            self.lstm_states = []
            self.rewards = []
            self.cvals = []
            self.programs_failed_indices = []
            self.programs_failed_initstates = []

            self.global_recursive_call = False

            # play an episode
            final_node, max_depth_reached = self._play_episode(self.root_node)
            final_node['selected'] = True

        # compute final task reward (with gamma penalization)
        reward = self.env.get_reward()
        if reward > 0:
            task_reward = reward * (self.gamma**final_node['depth'])
            if self.recursive_task and not self.global_recursive_call:
                # if recursive task but do not called itself, add penalization
                task_reward -= self.recursive_penalty
        else:
            task_reward = -1

        # Replace None rewards by the true final task reward
        self.rewards = list(map(lambda x: torch.FloatTensor([task_reward]) if x is None else torch.FloatTensor([x]), self.rewards))

        # end task
        self.env.end_task()

        return self.observations, self.programs_index, self.previous_actions, self.mcts_policies, \
               self.lstm_states, max_depth_reached, self.root_node, task_reward, self.clean_sub_executions, self.rewards, \
               self.programs_failed_indices, self.programs_failed_initstates, self.cvals

