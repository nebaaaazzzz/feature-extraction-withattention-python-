from torchvision.models.resnet import BasicBlock, ResNet
import torch
import torch.nn as nn

# groups for 1d convolution and  num_groups for group normalization
# ELA-T as kernel size = 5, groups = in_channels, num_group = 32;
# ELA-S is kernel size = 5, groups = in_channels/8, num_group = 16.
# ELA-B as kernel size = 7, groups = in_channels, num_group = 16;
# ELA-L is kernel size = 7, groups = in_channels/8, num_group = 16.

class EfficientLocalizationAttention(nn.Module):
    
    KERNEL_SIZE = 7
    NUMBER_GROUPS = 16
    IS_GROUPS = False
    
    def __init__(self, channel, kernel_size=KERNEL_SIZE, num_groups=NUMBER_GROUPS):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size,
                              padding=pad, groups=channel / 8 if self.IS_GROUPS else channel , bias=False)
        self.gn = nn.GroupNorm(num_groups, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Global average pooling across H and W
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)  # [b,c,h]
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)  # [b,c,w]

        # Apply convolution and normalization
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)  # [b,c,h,1]
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)  # [b,c,1,w]

        return x * x_h * x_w
    
    
class BasicBlockWithELA(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=7, num_groups=16):
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)

        self.ela = EfficientLocalizationAttention(planes, kernel_size=kernel_size, num_groups=num_groups)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Apply ELA attention
        out = self.ela(out)
        
        out += identity
        out = self.relu(out)


        return out
def get_resnet18(ela_kernelsize , ela_groups , ela_numgroup , with_attention=True  , **kwargs):
    """
    Creates a ResNet-18 model with ELA blocks.
    """
    model = None
    if with_attention : 
        BasicBlock.KERNEL_SIZE = ela_kernelsize
        BasicBlock.NUMBER_GROUPS = ela_numgroup
        BasicBlock.IS_GROUPS = ela_groups
        model = ResNet(BasicBlockWithELA, [2, 2, 2, 2], **kwargs)
    else :
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model    

