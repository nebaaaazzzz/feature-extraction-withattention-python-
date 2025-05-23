from torch import nn
import torch

# groups for 1d convolution and  num_groups for group normalization
# ELA-B as kernel size = 7, groups = in_channels, num_group = 16;
# ELA-T as kernel size = 5, groups = in_channels, num_group = 32;



# ELA-S is kernel size = 5, groups = in_channels/8, num_group = 16.
# ELA-L is kernel size = 7, groups = in_channels/8, num_group = 16.

class EfficientLocalizationAttention(nn.Module):
    
    KERNEL_SIZE = 7
    NUMBER_GROUPS = 16
    IS_GROUPS = False
    
    def __init__(self, channel, kernel_size=7, num_groups=16 , is_groups=False):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size,
                              padding=pad, groups=channel / 8 if is_groups else channel , bias=False)
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