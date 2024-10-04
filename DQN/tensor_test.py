from DQN.dqn_agent import DDQNAgent
import sys
sys.path.insert(1, 'd:\이도훈\Documents\Code\MINE_V2')
from minesweeper import Minesweeper


env = Minesweeper(5, 5, 5)

state_size = env.get_state().shape
action_size = (env.rows * env.cols)

agent = DDQNAgent(state_size, action_size)

state = env.get_state()
mask = -env.mask
action = agent.get_action(state, mask, 1.0)
row = int(action/env.cols)
col = int(action%env.cols)
next_state, reward, done = env.step(row, col, 0)
# Store the experience in the replay buffer and learn from it
agent.step(state, action, reward, next_state, done, mask, -env.mask)

state, action, reward, next_state, done, mask, next_mask = agent.replay_buffer.sample(1)
print(state)