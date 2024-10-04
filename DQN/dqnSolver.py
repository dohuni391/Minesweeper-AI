import torch
from dqn_agent import DDQNAgent
import sys
sys.path.insert(1, 'd:\이도훈\Documents\Code\MINE_V2')
from minesweeper import Minesweeper

class DQNBot():
    def __init__(self, env: Minesweeper, model):
        state_size = env.get_state().shape
        action_size = (env.rows * env.cols)
        self.agent = DDQNAgent(state_size, action_size)

        path = "./DQN/models/"+model+".pth"
        dict = torch.load(path)
        self.agent.qnetwork_local.load_state_dict(dict['current_state_dict'])

    def get_action(self, env: Minesweeper):
        state = env.get_state()
        mask = env.mask
        return self.agent.get_action(state, mask)