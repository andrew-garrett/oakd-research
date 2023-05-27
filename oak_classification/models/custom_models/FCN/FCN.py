#################################################
#################### IMPORTS ####################
#################################################


import torch.nn as nn

##############################################################
#################### CUSTOM MODEL CLASSES ####################
##############################################################


class View(nn.Module):
    """
    Custom nn.Module View class
    """
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)
    
class FCN(nn.Module):
    """
    Custom nn.Module Fully Convolutional Network Architecture, for classification
    """
    def __init__(self, c1=96, c2=192, d=0.5, num_classes=10):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.d = d
        self.num_classes = num_classes

        def fcn_block(in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=False):
            block = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,padding=padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            ]
            if dropout:
                block.append(nn.Dropout(d))
            return block

        self.model = nn.Sequential(
            nn.Dropout(0.2),
            *fcn_block(3, c1, 3, 1, 1),
            *fcn_block(c1, c1, 3, 1, 1),
            *fcn_block(c1, c1, 3, 2, 1, True),
            *fcn_block(c1, c2, 3, 1, 1),
            *fcn_block(c2, c2, 3, 1, 1),
            *fcn_block(c2, c2, 3, 2, 1, True),
            *fcn_block(c2, c2, 3, 1, 1),
            *fcn_block(c2, c2, 3, 1, 1),
            *fcn_block(c2, 10, 1, 1),
            nn.AvgPool2d(8),
            View(num_classes)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    FCN()