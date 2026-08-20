import torch
import torch.nn as nn
import torch.nn.functional as F

class ShapePriorBlock(nn.Module):
    def __init__(self, feat_channels, prior_channels=64):
        super().__init__()
        self.prior = nn.Parameter(torch.randn(1, prior_channels, 1, 1))
        self.self_update = nn.Sequential(
            nn.Conv2d(prior_channels, prior_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(prior_channels, prior_channels, kernel_size=3, padding=1)
        )
        self.cross_update = nn.Conv2d(feat_channels + prior_channels, feat_channels, kernel_size=1)

    def forward(self, skip_feat):
        B, C, H, W = skip_feat.shape
        prior_resized = F.interpolate(self.prior.expand(B, -1, -1, -1), size=(H, W), mode='bilinear', align_corners=False)
        prior_updated = self.self_update(prior_resized)
        fuse = torch.cat([skip_feat, prior_updated], dim=1)
        enhanced_skip = skip_feat + self.cross_update(fuse)
        return enhanced_skip, prior_updated


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        return self.pool(x2), x2


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(out_ch * 2, out_ch, 3, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        skip = skip[:, :, diffY//2: skip.size()[2]-diffY//2, diffX//2: skip.size()[3]-diffX//2]
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNetSPM(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.enc1 = UNetEncoderBlock(in_channels, 64)
        self.enc2 = UNetEncoderBlock(64, 128)
        self.enc3 = UNetEncoderBlock(128, 256)
        self.enc4 = UNetEncoderBlock(256, 512)

        self.spm1 = ShapePriorBlock(64)
        self.spm2 = ShapePriorBlock(128)
        self.spm3 = ShapePriorBlock(256)
        self.spm4 = ShapePriorBlock(512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=0), nn.ReLU(True),
            nn.Conv2d(1024, 1024, 3, padding=0), nn.ReLU(True)
        )
        self.dec4 = UNetDecoderBlock(1024, 512)
        self.dec3 = UNetDecoderBlock(512, 256)
        self.dec2 = UNetDecoderBlock(256, 128)
        self.dec1 = UNetDecoderBlock(128, 64)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, img):
        x, s1 = self.enc1(img)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        s1, _ = self.spm1(s1)
        s2, _ = self.spm2(s2)
        s3, _ = self.spm3(s3)
        s4, _ = self.spm4(s4)

        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        logits = self.out(x)
        return logits