from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=64)
        self.dout_1 = nn.Dropout2d(p=0.3)
        self.maxp_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=128)
        self.dout_2 = nn.Dropout2d(p=0.3)
        self.maxp_2 = nn.MaxPool2d(kernel_size=2)
        self.dense_1 = nn.Linear(in_features=7*7*128, out_features=64)
        self.dense_2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x))
        x = self.maxp_1(x)
        x = self.dout_1(x)
        x = F.leaky_relu(self.conv_2(x))
        x = self.maxp_2(x)
        x = self.dout_2(x)
        x = x.view(-1, 7*7*128)
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)

        return x
