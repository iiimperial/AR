import torch
import torch.nn as nn
from torchvision import models

class ResNetLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden=256):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.spatial_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden, num_layers=2, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, video_seq):
        B, T, C, H, W = video_seq.shape
        feats = []
        for t in range(T):
            frame = video_seq[:, t, :, :, :]
            f = self.spatial_encoder(frame).squeeze(-1).squeeze(-1)
            feats.append(f)
        seq_feat = torch.stack(feats, dim=1)
        lstm_out, _ = self.lstm(seq_feat)
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits