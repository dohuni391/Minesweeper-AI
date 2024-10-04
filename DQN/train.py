from DQN.dqn_agent import DDQNAgent
import sys
sys.path.insert(1, 'd:\이도훈\Documents\Code\MINE_V2')
from minesweeper import Minesweeper

# Create the environment
env = Minesweeper(9, 9, 10)

# Get the state and action sizes
state_size = env.get_state().shape
action_size = (env.rows * env.cols)

# Create the DDQN agent
agent = DDQNAgent(state_size, action_size)

# Set the number of episodes and the maximum number of steps per episode
num_episodes = 10000000
max_steps = env.rows*env.cols - env.mines

# Set the exploration rate
eps = eps_start = 0.95
eps_end = 0.0001
eps_decay = 0.95
reward_threshold = 1.0
reward_step = 0.1

# Set the statistics measure variables
MEASURE_STEP = 500
rewards = 0
wins = 0
efficiencies = 0
completions = 0
durations = 0

# Run the training loop
for i_episode in range(num_episodes):
    # Initialize the environment and the state
    env.reset()
    state = env.get_state()
    mask = env.mask

    # Run the episode
    for t in range(max_steps):
        # Select an action and take a step in the environment
        action = agent.get_action(state, mask, eps)
        row = int(action/env.cols)
        col = int(action%env.cols)
        next_state, reward, done = env.step(row, col, 0)
        # Store the experience in the replay buffer and learn from it
        agent.step(state, action, reward, next_state, done, mask, env.mask)
        # Update the state and the score
        state = next_state
        mask = env.mask
        rewards += reward
        # Break the loop if the episode is done or truncated
        if done != 0:
            # Save statistics
            if(done == 1):
                wins += 1
            efficiencies += env.efficiency
            durations += env.duration
            completions += env.completion_percentage
            break

    if(i_episode % MEASURE_STEP == 0):
        avg_reward = rewards/MEASURE_STEP
        win_rate = wins*100/MEASURE_STEP
        avg_completion = completions/MEASURE_STEP
        avg_efficiency = efficiencies/MEASURE_STEP
        avg_durations = durations/MEASURE_STEP
        agent.save_logs(i_episode, avg_reward, win_rate, avg_completion, avg_efficiency, avg_durations, eps)

        rewards = 0
        wins = 0
        efficiencies = 0
        completions = 0
        durations = 0

        # Update the exploration rate
        if(avg_reward>reward_threshold):
            eps = max(eps_end, eps_decay * eps)
            reward_threshold += reward_step

    if(i_episode % 50000 == 0):
        agent.save_checkpoints(i_episode, eps)