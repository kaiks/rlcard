''' DQN agent

The code is derived from https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py

Copyright (c) 2019 Matthew Judell
Copyright (c) 2019 DATA Lab at Texas A&M University
Copyright (c) 2016 Denny Britz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
import os
import pickle
#import copy


from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])

class DQNAgent(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.DQNAgent
    that depends on PyTorch instead of Tensorflow
    '''
    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=200,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=700000,
                 batch_size=32,
                 num_actions=2,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.00005,
                 device=None,
                 save_every=1000,
                 model_dir=None,
                 training_mode = True):

        '''
        Q-Learning algorithm for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
            replay_memory_size (int): Size of the replay memory
            replay_memory_init_size (int): Number of random experiences to sample when initializing
              the reply memory.
            update_target_estimator_every (int): Copy parameters from the Q estimator to the
              target estimator every N steps
            discount_factor (float): Gamma discount factor
            epsilon_start (float): Chance to sample a random action when taking an action.
              Epsilon is decayed over time and this is the start value
            epsilon_end (float): The final minimum value of epsilon after decaying is done
            epsilon_decay_steps (int): Number of steps to decay epsilon over
            batch_size (int): Size of batches to sample from the replay memory
            evaluate_every (int): Evaluate every N steps
            num_actions (int): The number of the actions
            state_space (list): The space of the state vector
            train_every (int): Train the network every X steps.
            mlp_layers (list): The layer number and the dimension of each layer in MLP
            learning_rate (float): The learning rate of the DQN agent.
            device (torch.device): whether to use the cpu or gpu
        '''
        self.use_raw = False
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.model_dir = model_dir
        self.save_every = save_every
        self.training_mode = training_mode
        
        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        if self.training_mode:
            self.debug = False
        else:
            self.debug = False #True

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # Create estimators
        self.q_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)
        self.target_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)

        # Create replay memory
        if self.training_mode:
            self.memory = Memory(replay_memory_size, batch_size)

    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        self.total_t += 1
        if self.training_mode:
            return
        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], list(next_state['legal_actions'].keys()), done)
        tmp = self.total_t - self.replay_memory_init_size
        if tmp>=0 and tmp%self.train_every == 0:
            self.train()

    def step(self, state):
        ''' Predict the action for generating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        if self.training_mode == False:
            return self.eval_step(self, state)[0]
        q_values = self.predict(state)
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)] # this calculation can probably be eliminated after t>epsilon end
        legal_actions = list(state['legal_actions'].keys())
        probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
        best_action_idx = legal_actions.index(np.argmax(q_values))
        probs[best_action_idx] += (1.0 - epsilon)
        # we can probably simplify this by choosing a value 0-1, if it's lt epsilon then pick an action at random, otherwise pick optimal
        # the current implementation might even be buggy for a large action space because we seem to be picking random actions with prob epsilon*len(action space size)
        action_idx = np.random.choice(np.arange(len(probs)), p=probs)

        return legal_actions[action_idx]

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        q_values = self.predict(state)
        best_action = np.argmax(q_values)

        # Debug:
        if self.debug:
            print('DQN agent legal actions', state['legal_actions'])
            print('DQN agent q_values', q_values)
            print('DQN agent best action', best_action)

        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return best_action, info

    def predict(self, state):
        ''' Predict the masked Q-values

        Args:
            state (numpy.array): current state

        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value
        '''
        
        q_values = self.q_estimator.predict_nograd(np.expand_dims(state['obs'], 0))[0]
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        #print('DQN agent legal actions', state['legal_actions'])
        legal_actions = list(state['legal_actions'].keys())
        masked_q_values[legal_actions] = q_values[legal_actions]
        # Debug info
        if self.debug:
            print('DQN agent legal actions', state['legal_actions'])
            print('DQN agent q_values', q_values)
            print('DQN agent masked_q_values', masked_q_values)

        return masked_q_values

    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        if self.training_mode == False:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()

        # Calculate best next actions using Q-network (Double DQN)
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
        masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
        best_actions = np.argmax(masked_q_values, axis=1)

        # Evaluate best next actions using Target-network (Double DQN)
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        # Perform gradient descent update
        state_batch = np.array(state_batch)

        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

        self.train_t += 1

        if self.train_t % self.save_every == 0:
            print("\nINFO - Saving model...")
            self.save(self.model_dir)
            print("\nINFO - Saved model.")

        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0 and self.model_dir is not None: 
# this could maybe be eliminated with the soft update set to 0.001            self.target_estimator.soft_update_from(self.q_estimator, 0.01)
            print("\nINFO - Copied model parameters to target network.")

    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        ''' Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        self.memory.save(state, action, reward, next_state, legal_actions, done)

    def set_device(self, device):
        self.device = device
        self.q_estimator.device = device
        self.target_estimator.device = device

    def debug(self, message):
        if self.debug:
            print(message)

    def current_training_parameters(self):
        return {
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'discount_factor': self.discount_factor,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'update_target_estimator_every': self.update_target_estimator_every,
            'train_every': self.train_every,
            'train_t': self.train_t,
            'num_actions': self.num_actions,
            'state_shape': self.state_shape,
            'total_t': self.total_t,
            'train_t': self.train_t,
            'mlp_layers': self.mlp_layers,
            'replay_memory_size': self.replay_memory_size,
            'replay_memory_init_size': self.replay_memory_init_size
        }
        

    def save(self, path_dir):
        ''' Save the model

        Args:
            path_dir (str): the path to the directory
        '''
        # save training parameters
        if self.training_mode:
            return
        
        with open(os.path.join(path_dir, 'params_dqn.bin'), 'wb') as f:
            pickle.dump(self.current_training_parameters(), f)
        
        # save memory
        self.memory.save_to_file(path_dir)

        # save model
        self.target_estimator.save(path_dir)

    def load_training_parameters(self, path_dir):
        ''' Load the training parameters from the directory

        Args:
            path_dir (str): the path to the directory
        '''
        print('INFO - Loading training parameters')
        with open(os.path.join(path_dir, 'params_dqn.bin'), 'rb') as f:
            params = pickle.load(f)
        print('INFO - Loaded training parameters from file:', params, '\n')
        self.epsilon_decay_steps = params['epsilon_decay_steps']
        self.epsilon_start = params['epsilon_start']
        self.epsilon_end = params['epsilon_end']
        self.discount_factor = params['discount_factor']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.update_target_estimator_every = params['update_target_estimator_every']
        self.train_every = params['train_every']
        self.train_t = params['train_t']
        self.num_actions = params['num_actions']
        self.state_shape = params['state_shape']
        self.total_t = params['total_t']
        self.train_t = params['train_t']
        self.mlp_layers = params['mlp_layers']
        self.replay_memory_size = params['replay_memory_size']
        self.replay_memory_init_size = params['replay_memory_init_size']
        
        if not self.training_mode:
            self.train_every = 9999999999999
            self.replay_memory_init_size = 0
            self.replay_memory_size = 0
            self.save_every = 9999999999999
            

    def reinitialize(self, params):
        print('INFO - Reinitializing helper objects')
        # print class of params variable
        print('INFO - params class:', type(params))
        
        # reinitialize the estimators
        self.q_estimator = Estimator(self.num_actions, self.learning_rate, self.state_shape, self.mlp_layers, self.device, params)
        self.target_estimator = Estimator(self.num_actions, self.learning_rate, self.state_shape, self.mlp_layers, self.device, params)

        self.epsilons = np.linspace(self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps)
        
        if self.training_mode:
            self.memory = Memory(self.replay_memory_size, self.batch_size)

    def load(self, path_dir):
        ''' Load the model

        Args:
            path_dir (str): the path to the directory
        '''
        print('INFO - Loading model from {}'.format(path_dir))
        
        self.load_training_parameters(path_dir)
        
        params = torch.load(os.path.join(path_dir, 'model.pth'))
        print('Params class:', type(params))
        print('INFO - Loaded model parameters from file:', params, '\n')
        self.reinitialize(params)
        del params
        
        if self.training_mode:
            print('INFO - Loading memory')
            self.memory.load_from_file(path_dir)
            print('INFO - Done loading model')

class Estimator(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.Estimator that
    uses PyTorch instead of Tensorflow.  All methods input/output np.ndarray.

    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    '''

    def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None, params=None):
        ''' Initilalize an Estimator object.

        Args:
            num_actions (int): the number output actions
            state_shape (list): the shape of the state space
            mlp_layers (list): size of outputs of mlp layers
            device (torch.device): whether to use cpu or gpu
        '''
        self.num_actions = num_actions
        self.learning_rate=learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.device = device

        # set up Q model and place it in eval mode
        qnet = EstimatorNetwork(num_actions, state_shape, mlp_layers)
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()

        
        if params is not None:
            self.qnet.load_state_dict(params)
        else:
            for p in self.qnet.parameters():
                if len(p.data.shape) > 1:
                    nn.init.xavier_uniform_(p.data)

        # set up loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        # set up optimizer
        self.optimizer =  torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

    def predict_nograd(self, s):
        ''' Predicts action values, but prediction is not included
            in the computation graph.  It is used to predict optimal next
            actions in the Double-DQN algorithm.

        Args:
          s (np.ndarray): (batch, state_len)

        Returns:
          np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
          action values.
        '''
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_as = self.qnet(s).cpu().numpy()
        return q_as

    def update(self, s, a, y):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)

        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target

        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.qnet.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # (batch, state_shape) -> (batch, num_actions)
        q_as = self.qnet(s)

        # (batch, num_actions) -> (batch, )
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # update model
        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss

    def soft_update_from(self, estimator, tau):
        ''' Soft update the model parameters from estimator

        Args:
            estimator (Estimator): the estimator to copy from
            tau (float): the soft update rate
        '''
        for target_param, param in zip(self.qnet.parameters(), estimator.qnet.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save(self, path):
        ''' Save the model to the path

        Args:
            path (str): path to save the model
        '''
        torch.save(self.qnet.state_dict(), path + '/model.pth')


class EstimatorNetwork(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        ''' Initialize the Q network

        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(EstimatorNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # build the Q network
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        fc = [nn.Flatten()]
        fc.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, s):
        ''' Predict action values

        Args:
            s  (Tensor): (batch, state_shape)
        '''
        return self.fc_layers(s)

    

class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, legal_actions, done):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done, legal_actions)
        self.memory.append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:       
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = random.sample(self.memory, self.batch_size)
        samples = tuple(zip(*samples))
        return tuple(map(np.array, samples[:-1])) + (samples[-1],)

    def save_to_file(self, path_dir):
        ''' Save the memory to a file

        Args:
            path_dir (str): the directory to save the memory
        '''
        with open(path_dir + '/memory.bin', 'wb') as f:
            pickle.dump(self.memory, f)

    def load_from_file(self, path_dir):
        ''' Load the memory from a file

        Args:
            path_dir (str): the directory to load the memory
        '''
        with open(path_dir + '/memory.bin', 'rb') as f:
            self.memory = pickle.load(f)