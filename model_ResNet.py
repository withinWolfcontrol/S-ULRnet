import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    """
    Basic Block for ResNet-18 and ResNet-34.
    It consists of two 3x3 convolutions.
    """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # First convolution with stride (can perform downsampling if stride > 1)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        
        # Second convolution
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # If input and output dimensions don't match, downsample the identity path
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add residual connection
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck Block for ResNet-50, 101, and 152.
    It uses a 1x1 -> 3x3 -> 1x1 structure to reduce parameters.
    
    Note:
    In the original paper, for the dotted residual branch (where dimensions change), 
    the stride of the first 1x1 conv layer is 2, and the second 3x3 conv layer is 1.
    
    However, in the official PyTorch implementation, the stride of the first 1x1 conv layer is 1, 
    and the second 3x3 conv layer is 2.
    
    This modification improves Top-1 accuracy by approximately 0.5%.
    Reference: ResNet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        # 1x1 conv: Squeeze channels (reduce dimension)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(width)
        
        # -----------------------------------------
        # 3x3 conv: Processing (stride is applied here in ResNet v1.5)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        
        # 1x1 conv: Unsqueeze channels (restore/expand dimension)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        # Initial convolution layer
        # Note: in_channels is set to 1 here (typically 3 for RGB images), likely for grayscale input
        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stacking the residual layers
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        
        self.tanh = nn.Tanh() # Not typically used in standard ResNet, likely custom addition
        
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling, output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        Constructs a sequence of residual blocks for a specific stage.
        """
        downsample = None
        # Create a downsample layer if stride != 1 or input channels != output channels
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # The first block in a layer handles the stride/downsampling
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        
        self.in_channel = channel * block.expansion

        # Subsequent blocks in the layer maintain the same channel dimensions
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            print(f"Feature map shape: {x.shape}") # Debugging: Print shape
            x = torch.flatten(x, 1)
            print(f"Flattened shape: {x.shape}")   # Debugging: Print shape
            pre = self.fc(x)
            
        return pre

# --- Factory Functions to Create Specific ResNet Models ---

def resnet18(num_classes=1000, include_top=True):
    """ Constructs a ResNet-18 model. """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    """ Constructs a ResNet-34 model. """
    # Weights: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    """ Constructs a ResNet-50 model. """
    # Weights: https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    """ Constructs a ResNet-101 model. """
    # Weights: https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    """ Constructs a ResNeXt-50 32x4d model. """
    # Weights: https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    """ Constructs a ResNeXt-101 32x8d model. """
    # Weights: https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


if __name__ == '__main__':
    # Initialize model (ResNet50 with 10 output classes)
    model = resnet50(num_classes=10, include_top=True)
    
    # Create dummy input data (Batch Size=16, Channels=1, Height=224, Width=224)
    # Note: Ensure channels match self.conv1 configuration (currently set to 1 in ResNet class)
    a = torch.randn(16, 1, 224, 224)
    
    # Forward pass
    pre = model(a)
    print(f"Output shape: {pre.shape}")

    # --- Legacy / Debugging Code below (commented out) ---
    # loss_fn = Physics_Loss_v3()
    # loss = loss_fn(pre_ps)
    # print(loss)
    
    # print(pre_s)
    
    # print(torch.cat([pre_p,pre_s],dim=1))
    # print(torch.cat([pre_p,pre_s],dim=1).shape)
    # print(c)
    # pre_p,pre_s = torch.split(pre_ps,1)
    # print(pre_p.shape,pre_s.shape)
    # predict_y = torch.max(pre_w, dim=1)[1]
    # print(predict_y)
    
    # test = torch.randn(7,4)
    # q = torch.split(test, [1,3],dim=-1)
    # print(q[0].shape,q[1].shape)