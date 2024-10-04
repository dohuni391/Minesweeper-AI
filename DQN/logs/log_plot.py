import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

colors = ["#ea5545", "#f46a9b", "#ef9b20", "#edbf33", "#ede15b", "#bdcf32", "#87bc45", "#27aeef", "#b33dc6"]

SMOOTH_VALUE = 50
def smooth(y):
    y_padded = np.pad(y, (SMOOTH_VALUE//2, SMOOTH_VALUE-1-SMOOTH_VALUE//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((SMOOTH_VALUE,))/SMOOTH_VALUE, mode='valid') 
    return y_smooth

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

SMOOTH = True
NORMALIZE = True

sns.set(style="darkgrid")
plt.xticks(size=14)
plt.yticks(size=14)
sns.despine(left=True,bottom=True)   
plt.xlabel("Number of Episodes")

plt.ion()

log_dnn = open("./DQN/logs/dqn_log.txt",'r')

episodes = []

rewards = []
losses = []
LRs = []
epsilons = []

wins = []
completions = []
efficiencies = []
durations = []

for line in log_dnn:
    splits = re.split("[:\t\n+]",line)
    episodes.append(int(splits[0]))
    rewards.append(float(splits[2]))
    wins.append(float(splits[4]))
    completions.append(float(splits[6]))
    efficiencies.append(float(splits[8]))
    durations.append(float(splits[10]))
    losses.append(float(splits[12]))
    LRs.append(float(splits[14][3:-2]))
    epsilons.append(float(splits[16]))

log_dnn.close()

episodes = np.asarray(episodes)/500

reward = np.asarray(rewards)
losses = np.asarray(losses)
LRs = np.asarray(LRs)
epsilons = np.asarray(epsilons)

wins = np.asarray(wins)
completions = np.asarray(completions)
efficiencies = np.asarray(efficiencies)
durations = np.asarray(durations)

if NORMALIZE:
    rewards = normalize(rewards)
    losses = normalize(losses)
    LRs = normalize(LRs)
    epsilons = normalize(epsilons)
    wins = normalize(wins)
    completions = normalize(completions)
    efficiencies = normalize(efficiencies)
    durations = normalize(durations)

if SMOOTH:
    rewards = smooth(rewards)
    losses = smooth(losses)
    LRs = smooth(LRs)
    epsilons = smooth(epsilons)
    wins = smooth(wins)
    completions = smooth(completions)
    efficiencies = smooth(efficiencies)
    durations = smooth(durations)

l1, = plt.plot(episodes, rewards, colors[0], label="Reward")
l2, = plt.plot(episodes, losses, colors[1], label="Loss")
# l3, = plt.plot(episodes, LRs, colors[2], label="Learning Rate")
# l4, = plt.plot(episodes, epsilons, colors[3], label="Epsilon")
l3, = plt.plot(episodes, wins, colors[4], label="Win Rate")
l4, = plt.plot(episodes, completions, colors[5], label="Completion Rate")
l5, = plt.plot(episodes, efficiencies, colors[6], label="Efficiency")
l6, = plt.plot(episodes, durations, colors[7], label="Duration")
plt.legend(handles=[l1, l2, l3, l4, l5, l6])
plt.show(block=True)