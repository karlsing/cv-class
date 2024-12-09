import torch
import torch.nn as nn

from torchvision.models import vit_b_16, ViT_B_16_Weights


class MyVITB16(nn.Module):
    def __init__(self):
        """
        Initializes network layers.
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        #######################################################################
        # Student code begins
        #######################################################################
        
        vit_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.conv_layers = vit_model
        self.conv_layers.requires_grad_(False) # freeze conv layers
        
        self.fc_layers = nn.Linear(1000, 15)
        
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
        x = x.repeat(1, 3, 1, 1)  # as vit accepts 3-channel color images
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
Temporary Done
"""