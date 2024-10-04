import tqdm
from dqnSolver import DQNBot
import sys
from minesweeper import Minesweeper

TEST_SIZE = 10000

env = Minesweeper(9, 9, 10)
model = 'ddqn_dnn750000'
DDQNBOT = DQNBot(env, model)
progress_bar = tqdm.tqdm(total=TEST_SIZE)

rewards = 0
win = 0
first_click_lose = 0
efficiency = 0
completion = 0
win_efficiency = 0
duration = 0

for _ in range(TEST_SIZE):
    env.reset()
    while env.done == 0:
        action = DDQNBOT.get_action(env)
        row = int(action/env.cols)
        col = int(action%env.cols)
        _, reward, _ = env.step(row, col)
        rewards += reward
        env.show()
        if env.done != 0:
            if env.clicks == 1:
                first_click_lose+=1
            break
    if env.done == 1:
        win+=1
        win_efficiency += env.efficiency
    efficiency += env.efficiency
    completion += env.completion_percentage
    duration += env.duration
    progress_bar.update(1)

progress_bar.close()

avg_reward = rewards/TEST_SIZE
win_rates, first_win_rate = (100 * win / TEST_SIZE), (100 * win / (TEST_SIZE - first_click_lose))
avg_completion = (completion / TEST_SIZE)
avg_efficiency, avg_win_efficiency =  (efficiency / TEST_SIZE), (win_efficiency / win+0.0000001)
avg_duration = (duration / TEST_SIZE)


print('\n[DQN RESULTS]\n'
        'Average Reward: {:.2f}% \n'
        'Win Rate: {}% \n'
        'Win Rate No First Lose: {:.2f}% \n'
        'Average Completion: {:.2f}% \n'
        'Average Efficiency: {:.2f}% \n'
        'Average Win Efficiency: {:.2f}% \n'
        'Average Duration: {:.2f} \n'
        .format(avg_reward, win_rates, first_win_rate, avg_completion, avg_efficiency, avg_win_efficiency, avg_duration)
    )