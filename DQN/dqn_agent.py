import random
import numpy as np
from numpy import float32
import torch
from torch import optim, FloatTensor
import torch.nn.functional as F
from dqn import DDQN, ReplayBuffer # type: ignore


class DDQNAgent():
    def __init__(self, state_size, action_size, learning_rate=5e-4, capacity=1000000, discount_factor=0.95, tau=1e-4, update_every=512, batch_size=2048):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.steps = 0

        self.qnetwork_local = DDQN(state_size, action_size)
        self.qnetwork_target = DDQN(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 1000, gamma = 0.8)
        self.replay_buffer = ReplayBuffer(capacity)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.log = open("./DQN/logs/dqn_log.txt",'w')
        self.loss = 0

    def step(self, state, action, reward, next_state, done, mask, next_mask):
        # Save experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done, mask, next_mask)

        # Learn every update_every steps
        self.steps += 1
        if self.steps % self.update_every == 0:
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample(self.batch_size)
                self.learn(experiences)

    def get_action(self, state, mask, eps=0.0):
        state = state.flatten()
        mask = mask.flatten()
        if random.random() > eps:
            state = FloatTensor(float32(np.reshape(state, (1, 1, 9, 9))))
            mask = FloatTensor(float32(mask))
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state, mask)
            self.qnetwork_local.train()
            action = action_values.max(1)[1].data[0].item()
        else:
            indices = np.nonzero(mask)[0]
            randno = random.randint(0,len(indices)-1)
            action = indices[randno]
        return action

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, mask, next_mask = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states, next_mask).max(1)[0]
        # Compute Q targets for current states 
        Q_targets = rewards + self.discount_factor * (Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states, mask).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.loss = loss.item() #Recording purpose

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_checkpoints(self, episode, eps):
        path = "./DQN/models/ddqn_dnn"+str(episode)+".pth"
        torch.save({
            'episode': episode,
            'current_state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict' : self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': eps
        }, path)

    def save_logs(self, episdoe, score, result, completion, efficiency, duration, eps):
        res = [
            str(episdoe),
            "\t Reward: ", str(score),
            "\t Win Rate: ", str(result),
            "\t Completion: ", str(round(completion)),
            "\t Efficiency: ", str(round(efficiency)),
            "\t Duration: ", str(round(duration)),
            "\t Loss: ", str(self.loss),
            "\t LR: ", str(self.lr_scheduler.get_last_lr()),
            "\t Epsilon: ", str(eps),
        ]
        log_line = " ".join(res)
        print(log_line)
        self.log.write(log_line+"\n") # type: ignore
        self.log.flush() # type: ignore