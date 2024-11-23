import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        """
        Constructor for SimpleNet class to define the layers and loss function.

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch's
        documention to understand what this means.
        """
        super(SimpleNet, self).__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        #######################################################################
        # Student code begins
        #######################################################################
        conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), padding=2)
        pool1 = nn.MaxPool2d((2, 2), 2)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(16, 64, kernel_size=(3, 3), padding=1)
        pool2 = nn.MaxPool2d((2, 2), 2)
        relu2 = nn.ReLU()
        adaptive = nn.AdaptiveMaxPool2d(1)
        
        self.conv_layers = nn.Sequential(
            conv1, pool1, relu1, conv2, pool2, relu2,
            adaptive, nn.Flatten(start_dim=1)
        )
        
        fc1 = nn.Linear(64, 32)
        act = nn.ReLU(inplace=True)
        fc2 = nn.Linear(32, 15)
        self.fc_layers = nn.Sequential(fc1, act, fc2)
        
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

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