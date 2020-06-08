## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Trying to build 5 convolution2d layers, 3 fully connected layers        
    
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        ## output size = (224 - 11 + 4) /4 + 1 = 55.25
        ## after the pool layer size becomes (64, 27, 27)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        # output size = (N + 2 * PADDING - (KERNEL_SIZE -1) - 1)/S + 1 = (64 - 2 * 2 - (5 - 1) -1) / 1 + 1 = 56
        ## after pool layer size becomes (192, 28, 28)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # maxpooling layers
        self.pool = nn.MaxPool2d(3,2)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        
        self.fc_drop = nn.Dropout()
        
        self.fc2 = nn.Linear(4096, 4096)
        
        ## 136 outputs
        self.fc3 = nn.Linear(4096, 136)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_drop(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc_drop(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x
