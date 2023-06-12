from torch import nn

__all__ = ['conv_4', 'conv_4_32F']


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

class Conv4(nn.Module):
    def __init__(self, mode=1):
        super(Conv4, self).__init__()
        if mode == 0:
            self.out_features = [32, 32, 32, 32]
        elif mode == 1:
            self.out_features = [64, 64, 64, 64]
        elif mode == 2:
            self.out_features = [64, 96, 128, 256]
        elif mode == 3:
            self.out_features = [96, 192, 384, 512]
        elif mode == 4:
            from models.STN import SpatialTransformer
            self.stnm = SpatialTransformer(3, (84, 84), 7, use_dropout=True)
            self.out_features = [64, 64, 64, 64]
        self.mode = mode
        self.conv1 = conv_block(3, self.out_features[0])
        self.conv2 = conv_block(self.out_features[0], self.out_features[1])
        self.conv3 = conv_block(self.out_features[1], self.out_features[2])
        self.conv4 = conv_block(self.out_features[2], self.out_features[3])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.mode == 4:
            x, _ = self.stnm(x)
        x = self.conv1(x)   # [_, _, 42, 42]
        
        import pdb
        pdb.set_trace()
        x = self.conv2(x)   # [_, _, 21, 21]
        x = self.conv3(x)   # [_, _, 10, 10]
        x = self.conv4(x)   # [_, _, 5, 5]

        return x

    def get_channel_num(self):

        return self.out_features

    def extract_feature(self, x, preReLU=False):

        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)

        return [feat1, feat2, feat3, feat4]

class Conv4_32F(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 32)
        self.conv3 = conv_block(32, 32)
        self.conv4 = conv_block(32, 32)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)   # [1, 64, 42, 42]
        x = self.conv2(x)   # [1, 64, 21, 21]
        x = self.conv3(x)   # [1, 64, 10, 10]
        x = self.conv4(x)   # [1, 64, 5, 5]

        return x

    def get_channel_num(self):

        return [32, 32, 32, 32]

    def extract_feature(self, x, preReLU=False):

        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)

        return [feat1, feat2, feat3, feat4]

# from : "Few-Shot Image Recognition by Predicting Parameters from Activations"
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        return x
       
# from : "Few-shot image recognition with knowledge transfer"
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # self.mxp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adp = nn.AdaptiveAvgPool2d((5, 5))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        # x = self.mxp4(x)
        x = self.adp(x)

        return x


def conv_4(mode=1):
    return Conv4(mode=mode)

def conv_4_32F():
    return Conv4_32F()

def conv_Qiao():
    return SimpleConvNet()

def conv_KTN():
    return ConvNet()
