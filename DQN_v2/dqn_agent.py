from collections import deque, namedtuple
import random
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from DQN_v2.dqn import DDQN

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 2048         # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 32        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    
    def __init__(self, input_shape):
        self.input_shape = input_shape

        self.qnetwork_local = DDQN(input_shape).to(device)
        self.qnetwork_target = DDQN(input_shape).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 1000, gamma = 0.8)
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0
        self.epsilon = 1

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.buffer.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.buffer) > BATCH_SIZE:
                experiences = self.buffer.sample()
                self.learn(experiences, GAMMA)
    
    def act(self, env, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            action = torch.argmax(action_values)
        else:
            return random.choice(self.action_size)
        
        operation = action // (env.num_of_rows * env.num_of_cols)
        position_in_grid = action % (env.num_of_rows * env.num_of_cols)
        row = position_in_grid // env.num_of_cols
        col = position_in_grid % env.num_of_rows
        return row, col, operation
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)      

    def soft_update(self, local_model, target_model, tau):
        #θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save(self, episode):
        path = "./DQN_v2/models/ddqn_"+str(episode)+".pth"
        torch.save({
            'episode': episode,
            'local_state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict' : self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load(self, model_name):
        path = "./DQN_v2/models/ddqn_" + model_name
        dict = torch.load(path, weights_only=True)
        self.qnetwork_local.load_state_dict(dict['local_state_dict'])
        self.qnetwork_target.load_state_dict(dict['target_state_dict'])
        self.optimizer.load_state_dict(dict['optimizer_state_dict'])
        self.scheduler.load_state_dict(dict['scheduler_state_dict'])
        self.epsilon = dict['epsilon']


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)