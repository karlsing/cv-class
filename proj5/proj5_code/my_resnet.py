import torch
import torch.nn as nn

from torchvision.models import resnet18


class MyResNet18(nn.Module):
    def __init__(self):
        """
        Initializes network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one.
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch's
            documention to understand what this means.

        Download pretrained ResNet using pytorch's API.

        Hint: see the import statements
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        #######################################################################
        # Student code begins
        #######################################################################

        res_model = resnet18()
        block_list = list(res_model.children())[:-1] # remove last fc layer in resnet18
        self.conv_layers = nn.Sequential(*block_list, nn.Flatten())
        self.conv_layers.requires_grad_(False) # freeze conv layers
        
        fc1 = nn.Linear(512, 128)
        act1 = nn.ReLU()
        fc2 = nn.Linear(128, 15)
        self.fc_layers = nn.Sequential(fc1, act1, fc2)
        
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

        #######################################################################
        # Student code ends
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net, duplicating grayscale channel to
        3 channels.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images

        Returns:
            y: tensor of shape (N,num_classes) representing the output
                (raw scores) of the network. Note: we set num_classes=15
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
        #######################################################################
        # Student code begins
        #######################################################################

        x = self.conv_layers(x)
        model_output = self.conv_layers(x)

        #######################################################################
        # Student code ends
        #######################################################################
        return model_output

"""
Temporary Done
"""