import torch
import torch.nn as nn
from torch import FloatTensor,LongTensor
from collections import deque
import numpy as np
from numpy import float32
import random

class DDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQN, self).__init__()

        self.state_size = state_size

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding="same"),
            nn.ReLU(),
        )

        self.flatten_size = 256 * state_size[0] * state_size[1]
        
        self.advantage = nn.Sequential(
            nn.Linear(self.flatten_size,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    ### This is important, masks invalid actions
    def masked_softmax(self, vec, valid, dim=1, epsilon=1e-5):
        valid = -valid
        exps = torch.exp(vec)
        masked_exps = exps * valid.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps/masked_sums)
        
    def forward(self, x, mask):
        x=x/8
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        advantage = self.masked_softmax(self.advantage(x), mask)
        value = self.masked_softmax(self.value(x), mask)
        return value + advantage - advantage.mean()

class ReplayBuffer():
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state.flatten(), action, reward, next_state.flatten(), done!=0, mask.flatten(), next_mask.flatten()))

    def sample(self,batch_size):
        state, action, reward, next_state, done, mask, next_mask = zip(*random.sample(self.buffer, batch_size))
        state = FloatTensor(float32(np.reshape(state, (batch_size, 1, 9, 9))))
        action = LongTensor(float32(action))
        next_state = FloatTensor(float32(np.reshape(next_state, (batch_size, 1, 9, 9))))
        reward = FloatTensor(reward)
        done = FloatTensor(done)
        mask = FloatTensor(float32(mask))
        next_mask = FloatTensor(float32(next_mask))
        return state, action, reward, next_state, done, mask, next_mask
    
    def __len__(self):
        return len(self.buffer)