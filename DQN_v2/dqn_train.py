import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import numpy as np
from minesweeper_v2 import Minesweeper
from dqn_agent import DQNAgent

env = Minesweeper(16, 30, 99)
agent = DQNAgent(input_shape=env.get_state().shape)

n_episodes = 10000
max_steps = env.num_of_rows * env.num_of_cols - env.num_of_mines

eps_decay = 0.995
eps_end = 0

total_reward = 0
rewards_list = []
win_counts = []

model_name = 'dqn_model'
if os.path.isfile(model_name):
    agent.load(model_name)
    print("Loaded model from", model_name)

for i_episode in range(1, n_episodes+1):
    state = env.newGame()

    for t in range(max_steps):
        agent.t_step += 1
        action = agent.act(env, state)
        next_state, reward, done = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done != 0:
            if env.status == 1:
                win_counts.append(1)
            else:
                win_counts.append(0)
            break

    rewards_list.append(total_reward)
    agent.epsilon = max(eps_end, eps_decay * agent.epsilon)
    if i_episode % 100 == 0:
        avg_reward = np.mean(rewards_list[-100:])
        win_percentage = np.sum(win_counts[-100:])
        print(f'Episode {i_episode}, Average Reward: {avg_reward:.2f}, Win Percentage: {win_percentage:.2f}%')
    if i_episode % 1000 == 0:
        agent.save(i_episode)
