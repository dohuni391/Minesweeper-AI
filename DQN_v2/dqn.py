import torch
import torch.nn as nn

class DDQN(nn.Module):
    def __init__(self, input_shape, seed):
        super(DDQN, self).__init__()

        self.seed = torch.manual_seed(seed)

        # Convolutional layers
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
        )
        
        # Common fully connected layer after convolutional layers
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 16 * 30, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Value and advantage streams
        self.value = nn.Linear(256, 1)  # Output is a single scalar (value of state)
        self.advantage = nn.Linear(256, 2 * input_shape[0] * input_shape[1])

    def forward(self, x):
        x = x/8
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()
