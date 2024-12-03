import torch
from torch import nn
import timm

class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder='tf_efficientnet_b7_ns'):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.encoder.num_features, 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x