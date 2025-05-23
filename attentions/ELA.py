from torch import nn
import torch

# groups for 1d convolution and  num_groups for group normalization
# ELA-B as kernel size = 7, groups = in_channels, num_group = 16;
# ELA-T as kernel size = 5, groups = in_channels, num_group = 32;



# ELA-S is kernel size = 5, groups = in_channels/8, num_group = 16.
# ELA-L is kernel size = 7, groups = in_channels/8, num_group = 16.


# --- 1. Efficient Localization Attention (ELA) Module ---
class EfficientLocalizationAttention(nn.Module):
    """
    Efficient Localization Attention (ELA) module.
    Configurable for variants like ELA-S.
    """
    def __init__(self, channel, kernel_size=7,num_groups=16, group_setting="channel", ):
        super().__init__()
        pad = kernel_size // 2

        if group_setting == "channel":
            actual_conv_groups = channel
        elif group_setting == "channel/8":
            if channel % 8 != 0:
                # Try to find the closest valid group count if direct division fails
                # This might happen if channel is small, e.g. 64 not divisible by 8 for groups, but by channel for conv_groups
                print(f"Warning: Channel ({channel}) for ELA conv_groups is not perfectly divisible by 8. Using channel // 8 = {channel//8}.")
                actual_conv_groups = channel // 8
                if actual_conv_groups == 0 : actual_conv_groups = 1 # Ensure groups is at least 1
            else:
                actual_conv_groups = channel // 8
        else:
            try:
                actual_conv_groups = int(group_setting)
            except ValueError:
                raise ValueError(f"Invalid group_setting: {group_setting}")

        if channel % actual_conv_groups != 0:
             # Fallback if groups don't divide channels, use 1 group (standard conv)
             print(f"Warning: Channel ({channel}) not divisible by actual_conv_groups ({actual_conv_groups}). Defaulting to 1 conv group for this ELA block.")
             actual_conv_groups = 1

        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=pad, groups=actual_conv_groups, bias=False)

        # Adjust num_groups if channel is not divisible
        if channel % num_groups != 0:
            possible_gn_groups = [g for g in range(1, min(num_groups, channel) + 1) if channel % g == 0]
            if not possible_gn_groups:
                 actual_num_groups = 1 # Smallest possible group
                 print(f"Warning: Channel ({channel}) has no common divisor with num_groups up to {num_groups}. Using 1 group for GroupNorm.")
            else:
                actual_num_groups = max(possible_gn_groups) # Largest valid divisor
            if actual_num_groups != num_groups:
                print(f"Warning: Channel ({channel}) is not divisible by num_groups ({num_groups}). "
                      f"Using num_groups = {actual_num_groups} instead for this ELA block.")
        else:
            actual_num_groups = num_groups

        self.gn = nn.GroupNorm(actual_num_groups, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h_pooled = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w_pooled = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        att_h = self.conv(x_h_pooled)
        att_h = self.gn(att_h)
        att_h = self.sigmoid(att_h).view(b, c, h, 1)

        att_w = self.conv(x_w_pooled)
        att_w = self.gn(att_w)
        att_w = self.sigmoid(att_w).view(b, c, 1, w)

        return x * att_h * att_w