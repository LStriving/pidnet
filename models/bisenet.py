import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
        activation_hub = {'relu': nn.ReLU,             'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU,    'prelu': nn.PReLU,
                          'celu': nn.CELU,              'elu': nn.ELU, 
                          'hardswish': nn.Hardswish,    'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU,              'glu': nn.GLU, 
                          'selu': nn.SELU,              'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid,        'softmax': nn.Softmax, 
                          'tanh': nn.Tanh,              'none': nn.Identity,
                        }
                        
        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')
        
        self.activation = activation_hub[act_type](**kwargs)
        
    def forward(self, x):
        return self.activation(x)


# Regular convolution with kernel size 3x3
def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                    padding=1, bias=bias)


# Regular convolution with kernel size 1x1, a.k.a. point-wise convolution
def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, 
                    padding=0, bias=bias)

# Regular convolution -> batchnorm -> activation
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, 
                    bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation
            
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )

class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type, hid_channels=128):
        super(SegHead, self).__init__(
            ConvBNAct(in_channels, hid_channels, 3, act_type=act_type),
            conv1x1(hid_channels, num_class)
        )

class ResNet(nn.Module):
    # Load ResNet pretrained on ImageNet from torchvision, see
    # https://pytorch.org/vision/stable/models/resnet.html
    def __init__(self, resnet_type, pretrained=True):
        super(ResNet, self).__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

        resnet_hub = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50,
                        'resnet101':resnet101, 'resnet152':resnet152}
        if resnet_type not in resnet_hub:
            raise ValueError(f'Unsupported ResNet type: {resnet_type}.\n')
        
        # pretrained = pretrained
        resnet = resnet_hub[resnet_type](weights='ResNet18_Weights.IMAGENET1K_V1')
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)       # 2x down
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 4x down
        x1 = self.layer1(x)
        x2 = self.layer2(x1)      # 8x down
        x3 = self.layer3(x2)      # 16x down
        x4 = self.layer4(x3)      # 32x down

        return x1, x2, x3, x4

class SpatialPath(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type):
        super(SpatialPath, self).__init__(
            ConvBNAct(in_channels, out_channels, 3, 2, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, 2, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, 2, act_type=act_type),
        )


class ContextPath(nn.Module):
    def __init__(self, out_channels, backbone_type, act_type):
        super(ContextPath, self).__init__()
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [256, 512] if ('18' in backbone_type) or ('34' in backbone_type) else [1024, 2048]
        else:
            raise NotImplementedError()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.arm_16 = AttentionRefinementModule(channels[0])
        self.arm_32 = AttentionRefinementModule(channels[1])

        self.conv_16 = conv1x1(channels[0], out_channels)
        self.conv_32 = conv1x1(channels[1], out_channels)

    def forward(self, x):
        _, _, x_16, x_32 = self.backbone(x)
        x_32_avg = self.pool(x_32)
        x_32 = self.arm_32(x_32)
        x_32 += x_32_avg
        x_32 = self.conv_32(x_32)
        x_32 = F.interpolate(x_32, scale_factor=2, mode='bilinear', align_corners=True)

        x_16 = self.arm_16(x_16)
        x_16 = self.conv_16(x_16)
        x_16 += x_32
        x_16 = F.interpolate(x_16, scale_factor=2, mode='bilinear', align_corners=True)
        
        return x_16


class AttentionRefinementModule(nn.Module):
    def __init__(self, channels):
        super(AttentionRefinementModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBNAct(channels, channels, 1, act_type='sigmoid')

    def forward(self, x):
        x_pool = self.pool(x)
        x_pool = x_pool.expand_as(x)
        x_pool = self.conv(x_pool)
        x = x * x_pool

        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, act_type=act_type)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Sequential(
                                conv1x1(out_channels, out_channels),
                                nn.ReLU(),
                                conv1x1(out_channels, out_channels),
                                nn.Sigmoid(),
                            )

    def forward(self, x_low, x_high):
        x = torch.cat([x_low, x_high], dim=1)
        x = self.conv1(x)

        x_pool = self.pool(x)
        x_pool = x_pool.expand_as(x)
        x_pool = self.conv2(x_pool)

        x_pool = x * x_pool
        x = x + x_pool

        return x

class BiSeNetv1(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='resnet18', act_type='relu',):
        super(BiSeNetv1, self).__init__()
        self.spatial_path = SpatialPath(n_channel, 128, act_type=act_type)
        self.context_path = ContextPath(256, backbone_type, act_type=act_type)
        self.ffm = FeatureFusionModule(384, 256, act_type=act_type)
        self.seg_head = SegHead(256, num_class, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]
        if size == (1080, 1920):
            x = F.interpolate(x, [1024,1536], mode='bilinear', align_corners=True)
        x_s = self.spatial_path(x)
        x_c = self.context_path(x)
        x = self.ffm(x_s, x_c)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':
    model = BiSeNetv1(11)
    model #.to('cuda:0')
    input = torch.randn(1, 3, 1080, 1920) #.cuda()
    output = model(input)
    print(output.shape)
