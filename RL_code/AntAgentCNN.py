import torch
import torch.nn as nn
import torch.nn.functional as F

'''Input layer takes a low-res image (84x84 pixels here but we can change it)
3 convolutional layers with relu activation + flattening layer + fully connected layer and output layer. 
This should output the Q value for 3 possible actions (going right, going forward, going left)''' 

class AntAgentCNN(nn.Module):
    def __init__(self, input_channels=1, num_actions=3): 
        super(AntAgentCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.fc1 = nn.Linear(3072, 512)  # <-- We adjust based on input image size
        self.out = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x / 255.0  # Normalize pixel values
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.out(x)
