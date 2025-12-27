"""
SSSTFN Model：
Implements the fusion of spatial features extracted from
the second-to-last IRB block of the MobileNetV2 backbone
network with temporal features from the LSTM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# LSTM
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)


# STFAM
class SimpleAttention(nn.Module):
    def __init__(self, img_dim, time_dim, fused_dim):
        super(SimpleAttention, self).__init__()
        self.query = nn.Linear(time_dim, fused_dim)
        self.key = nn.Linear(img_dim, fused_dim)
        self.value = nn.Linear(img_dim, fused_dim)

    def forward(self, img_features, time_features):
        query = self.query(time_features).unsqueeze(1)  # (batch_size, 1, fused_dim)
        key = self.key(img_features)  # (batch_size, H*W, fused_dim)
        value = self.value(img_features)  # (batch_size, H*W, fused_dim)

        attention_scores = torch.bmm(query, key.transpose(1, 2))  # (batch_size, 1, H*W)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 1, H*W)
        attention_output = torch.bmm(attention_weights, value)  # (batch_size, 1, fused_dim)
        return attention_output.squeeze(1)  # (batch_size, fused_dim)

# SSSTFN
class SSSTFN(nn.Module):
    def __init__(self, num_classes=7, img_channels=96):
        super(SSSTFN, self).__init__()

        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet_v2.features

        self.feature_extractor_part1 = nn.Sequential(*list(self.features.children())[:13])
        self.feature_extractor_part2 = nn.Sequential(*list(self.features.children())[13:])

        self.feature_channels = img_channels

        self.lstm_feature_extractor = LSTMFeatureExtractor(input_dim=2, hidden_dim=self.feature_channels)

        self.attention = SimpleAttention(img_dim=self.feature_channels, time_dim=self.feature_channels,
                                         fused_dim=self.feature_channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)
    def forward(self, img_data, time_data):
        x = self.feature_extractor_part1(img_data)
        B, C, H, W = x.shape

        img_features = x.view(B, C, -1).permute(0, 2, 1)

        time_features = self.lstm_feature_extractor(time_data)

        fused_features = self.attention(img_features, time_features)

        fused_features = fused_features.unsqueeze(-1).unsqueeze(-1)
        broadcasted_fused_features = fused_features.expand(-1, -1, H, W)

        x = x + broadcasted_fused_features

        x = self.feature_extractor_part2(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = SSSTFN()
    img_input = torch.randn(16, 3, 224, 224)
    time_input = torch.randn(16, 72, 2)
    output = model(img_input, time_input)
    print(output.shape)
