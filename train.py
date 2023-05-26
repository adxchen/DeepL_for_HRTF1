import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary


# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate


# 自定义神经网络类
class SubNetA(nn.Module):
    def __init__(self):
        super(SubNetA, self).__init__()
        #  sub_a
        self.Alinear = nn.Sequential(
            nn.Linear(19, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
        )

    def forward(self, x_anthro):
        x = self.Alinear(x_anthro)

        return  x    # return x for visualization

class SubNetB(nn.Module):
    def __init__(self):
        super(SubNetB, self).__init__()
        # sub_b
        self.Bconv1 = nn.Sequential(         # input shape (1, 64, 64)
            # 卷积层:(1, 64, 64) -> (16, 64, 64)
            nn.Conv2d(
                in_channels=1,              # input height，灰度图片为1，rgb图片为3
                out_channels=16,            # n_filters:16
                kernel_size=(3,3),              # filter size: 3*3
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 64, 64)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 32, 32)
        )
        self.Bconv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),     # output shape (16, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (16, 7, 7)
        )
        self.B_out = nn.Sequential(
            nn.Linear(16 * 7 * 7, 8),  # fully connected layer
            nn.ReLU()
        )

    def forward(self, x_ear):
        y = self.Bconv1(x_ear)
        y = self.Bconv2(y)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        return y    # return x for visualization


class SubNetC(nn.Module):
    def __init__(self):
        super(SubNetC, self).__init__()
        # sub_c
        self.Clinear = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 200),
            nn.ReLU(),
        )



    def forward(self, c_input):
        output = self.Clinear(c_input)

        return output


sub_a = SubNetA()
print(sub_a)  # net architecture
summary(sub_a, input_size=(1, 1, 19))
