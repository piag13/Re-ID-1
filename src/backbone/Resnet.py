from torchvision import models
import torch.nn as nn


class ResNet50Embedding(nn.Module):
    def __init__(self):
        super(ResNet50Embedding, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # Remove the classification layer (fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer

    def forward(self, x):
        x = self.backbone(x)  # Shape: [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [B, 2048]
        return x