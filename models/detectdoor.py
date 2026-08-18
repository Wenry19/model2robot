
import torch
import torch.nn as nn

from models.conv_block import ConvBlock
from models.res_block import ResBlock

class Detectdoor(torch.nn.Module):

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug

        self.conv_block_1 = ConvBlock(3, 32)
        self.pool_1 =  nn.MaxPool2d(2)

        self.conv_block_2 = ConvBlock(32, 64)
        self.res_block_1 = ResBlock(64)

        self.conv_block_3 = ConvBlock(64, 128)
        self.res_block_2 = ResBlock(128)

        self.pool_2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):

        self._debug_shape(x)
        
        x = self.conv_block_1(x)
        self._debug_shape(x)
        x = self.pool_1(x)
        self._debug_shape(x)

        x = self.conv_block_2(x)
        self._debug_shape(x)
        x = self.res_block_1(x)
        self._debug_shape(x)

        x = self.conv_block_3(x)
        self._debug_shape(x)
        x = self.res_block_2(x)
        self._debug_shape(x)

        x = self.pool_2(x)
        self._debug_shape(x)
        x = torch.flatten(x, 1)
        self._debug_shape(x)
        x = self.fc(x)
        self._debug_shape(x)

        return x

    def _debug_shape(self, x):
        if self.debug:
            print(x.shape)

if __name__ == "__main__":

    classifier = Detectdoor(debug=True)

    x = torch.randn(1, 3, 224, 224)

    classifier.forward(x)
