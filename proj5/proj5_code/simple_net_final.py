import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Constructor for SimpleNetFinal class to define the layers and loss
        function.

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch's
        documention to understand what this means.
        """
        super(SimpleNetFinal, self).__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        #######################################################################
        # Student code begins
        #######################################################################
        conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), padding=2) # 64*64
        bn1   = nn.BatchNorm2d(16)
        relu1 = nn.ReLU()
        pool1 = nn.MaxPool2d((2, 2), 2) # 32 * 32
        conv2 = nn.Conv2d(16, 64, kernel_size=(3, 3), padding=1) # 32 * 32
        bn2   = nn.BatchNorm2d(64)
        relu2 = nn.ReLU()
        pool2 = nn.MaxPool2d((2, 2), 2) # 16 * 16
        conv3 = nn.Conv2d(64, 256, kernel_size=(3, 3), padding=1) # 16 * 16
        bn3   = nn.BatchNorm2d(256)
        relu3 = nn.ReLU()
        pool3 = nn.MaxPool2d((2, 2), 2) # 8 * 8
        conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1) # 8 * 8
        relu4 = nn.ReLU()
        pool4 = nn.MaxPool2d((2, 2), 2) # 4 * 4
        dropout = nn.Dropout(p=0.3)
        
        fc1  = nn.Linear(512*4*4, 1024)
        act1 = nn.ReLU()
        fc2  = nn.Linear(1024, 256)
        act2 = nn.ReLU()
        fc3  = nn.Linear(256, 15)
        
        self.conv_layers = nn.Sequential(
            conv1, bn1, relu1, pool1,
            conv2, bn2, relu2, pool2,
            conv3, bn3, relu3, pool3,
            conv4, relu4, pool4,
            dropout,
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            fc1, act1, 
            fc2, act2, 
            fc3
        )
        
        self.loss_criterion = nn.CrossEntropyLoss()

        #######################################################################
        # Student code ends
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the network.

        Args:
            x: the (N,C,H,W) input images

        Returns:
            y: the (N,15) output (raw scores) of the net
        """
        model_output = None
        #######################################################################
        # Student code begins
        #######################################################################

        x = self.conv_layers(x)
        model_output = self.fc_layers(x)

        #######################################################################
        # Student code ends
        #######################################################################
        return model_output

"""
Already Done
"""