import torch
import torch.nn as nn

class DDQN(nn.Module):
    def __init__(self, input_shape):
        super(DDQN, self).__init__()

        # Convolutional layers
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
        )
        
        # Common fully connected layer after convolutional layers
        self.fc2 = nn.Sequential(
            nn.Linear(64 * 16 * 30, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * input_shape[0] * input_shape[1])
        )

    def forward(self, x):
        x[x == -1] = 9  # Map covered tiles to a value higher than 8
        x[x == -2] = 10  # Map flagged tiles to a value higher than 9
        x = x / 10       # Normalize everything between 0 and 1
        x = self.fc1(x)
        x = self.fc2(x)
        return x
