from pickle import NONE
import torch.nn as nn
import torch
from src.cvutils.cnn.bricks.registry import NORM_LAYERS



class ClusterNorm2d(nn.Module):
    def __init__(self, num_features, num_classes=None, label=None):
        super().__init__()
        if num_classes is None:
            num_classes = 100
        else:
            num_classes = num_classes
            
        self.y = None

        cluster_bns = []
        for _ in num_classes:
            cluster_bns.append(nn.BatchNorm2d(num_features, affine=False))
        self.cluster_bn = nn.ModuleList(cluster_bns)
        
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=True)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02
        )  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma, beta = self.embed(self.y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1
        )
        return out