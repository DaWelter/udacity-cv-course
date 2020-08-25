import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


def make_stage(filters_in, kernel):
    return nn.Sequential(
            nn.Conv2d(filters_in, filters_in*2, kernel),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
    )

# With max pooling
# Epoch: 200, Batch: 50, Avg. Loss: 0.00589
# lr=1.e-4
# And with avg pooling
# Loss over test set = 0.00452

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Output 68x2
        # Input 192
        self.initial_stage = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )

        self.convnet = nn.Sequential(
            make_stage(32, 3), # 92+2 = 94
            make_stage(64, 3), # 44+2 = 46
            make_stage(128, 3), # 20+2 = 22
            make_stage(256, 3), # 8+2 = 10
        )

        self.dense1 = nn.Linear(512*4*4, 1024)
        self.drop1 = torch.nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(1024, 1024)
        self.drop2 = torch.nn.Dropout(p=0.2)
        self.dense3 = nn.Linear(1024, 68*2)


    def forward(self, x):
        #print (x.shape)
        x = self.initial_stage(x)
        #print (x.shape)
        x = self.convnet(x)
        #print (x.shape)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(F.relu(x))
        x = self.drop2(x)
        x = self.dense3(F.relu(x))
        coordinates = x.view(x.size(0), 68, 2)
        return coordinates
