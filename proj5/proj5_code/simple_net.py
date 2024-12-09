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
        #input size: 1*64*64
        conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5), padding=0) # 10*60*60
        pool1 = nn.MaxPool2d((3, 3), 3) # 10*20*20
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5), padding=0) # 20*16*16
        pool2 = nn.MaxPool2d((3, 3), 3) # 20*5*5
        relu2 = nn.ReLU()
        
        self.conv_layers = nn.Sequential(
            conv1, pool1, relu1, conv2, pool2, relu2,
            nn.Flatten(start_dim=1)
        )
        
        fc1 = nn.Linear(20*5*5, 100)
        act1 = nn.ReLU()
        fc2 = nn.Linear(100, 15)
        self.fc_layers = nn.Sequential(fc1, act1, fc2)
        
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