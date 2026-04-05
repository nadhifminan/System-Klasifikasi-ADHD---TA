import torch
import torch.nn as nn

class SegToClassifier(nn.Module):
    def __init__(self, seg_net, input_channels=1, target_shape=(64,64,64), num_classes=2):
        super().__init__()

        # Jika DataParallel
        if isinstance(seg_net, nn.DataParallel):
            seg_net = seg_net.module

        # Backbone MedicalNet
        self.conv1 = seg_net.conv1
        self.bn1 = seg_net.bn1
        self.relu = seg_net.relu
        self.maxpool = seg_net.maxpool
        self.layer1 = seg_net.layer1
        self.layer2 = seg_net.layer2
        self.layer3 = seg_net.layer3
        self.layer4 = seg_net.layer4

        self.pool = nn.AdaptiveAvgPool3d(1)

        # Infer feature dim (AMAN & IDENTIK DENGAN TRAINING)
        with torch.no_grad():
            dummy = torch.randn(1, input_channels, *target_shape)
            feat = self._forward_backbone(dummy)
            in_features = self.pool(feat).flatten(1).shape[1]

        self.fc = nn.Linear(in_features, num_classes)

    def _forward_backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self._forward_backbone(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
