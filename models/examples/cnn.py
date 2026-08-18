
import torch
import torch.nn.functional as F

class LeNet(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # input images: 1x32x32
        # 1 input image channel,
        # 6 output channels,
        # 5x5 square convolution kernel
        # we have 6 kernels (with different weights) of size 5x5
        # these kernels produce an output of 6 channels
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)

        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Vocabulary:
# Kernel = a small 2D matrix of weights applied to one input channel
# Filter = a collection of kernels that spans all input channels
#           and produces one output feature map

# self.conv = torch.nn.Conv2d(3, 6, 5)

# Input image:
# RGB image with 3 channels and spatial size 224x224
# PyTorch format: (batch, channels, height, width)
# Shape: (1, 3, 224, 224)

# The convolution has:
# 6 different filters (because out_channels=6)
# Each filter contains 3 kernels:
#   - one 5x5 kernel for the R channel
#   - one 5x5 kernel for the G channel
#   - one 5x5 kernel for the B channel

# Weight tensor shape:
# (out_channels, in_channels, kernel_height, kernel_width)
# (6, 3, 5, 5)

# Each filter produces one output feature map:
# Filter 1 -> feature map 1
# Filter 2 -> feature map 2
# ...
# Filter 6 -> feature map 6

# Output spatial size formula:
#
# output = (input_size - kernel_size + 2*padding) / stride + 1
#
# For this convolution:
# input_size = 224
# kernel_size = 5
# padding = 0
# stride = 1
#
# output = (224 - 5 + 2*0) / 1 + 1
#        = 220
#
# Therefore:
# Input:  (1, 3, 224, 224)
# Output: (1, 6, 220, 220)

# PyTorch uses channels-first ordering:
# (batch, channels, height, width)
