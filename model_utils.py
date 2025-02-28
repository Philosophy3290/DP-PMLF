import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self,input_size,num_classes):
        super(LinearModel,self).__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size,num_classes)
    
    def forward(self,feature):
        output = self.linear(feature.view(-1, self.input_size))
        return output

class CNN5(torch.nn.Module):

    def __init__(self, num_classes = 10, normalization = False):
        super(CNN5, self).__init__()
        self.normalization = normalization
        if self.normalization:
            self.gn0 = nn.GroupNorm(num_groups=16, num_channels=32)
            self.gn1 = nn.GroupNorm(num_groups=16, num_channels=64)
            self.gn2 = nn.GroupNorm(num_groups=16, num_channels=128)
            self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256)
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 100, kernel_size=3, stride=1, padding=1),
            torch.nn.AvgPool2d(kernel_size=4, stride=4),
        )

    def forward(self, x):
        x = self.layer0(x)
        if self.normalization:
            x = self.gn0(x)
        x = self.layer1(x)
        if self.normalization:
            x = self.gn1(x)
        x = self.layer2(x)
        if self.normalization:
            x = self.gn2(x)
        x = self.layer3(x)
        if self.normalization:
            x = self.gn3(x)
        y = self.layer4(x).view(x.size(0), -1)
        return y