from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet

from attentions.ELA import EfficientLocalizationAttention
from attentions.SE import SEAttention    
from attentions.CA import CoordAtt
from attentions.ECA import ECAAttention
from attentions.BAM import BAMBlock    
from attentions.CBAM import CBAMBlock
from attentions.A2 import DoubleAttention


class BasicBlockWithAttention(BasicBlock):
    
    expansion = 1
    ATTENTION_TYPE = None
    KERNEL_SIZE = 7
    NUMBER_GROUPS = 16
    GROUP_SETTING = "channel"
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,):
        
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        
        if(self.ATTENTION_TYPE == 'SE'):
            self.attention = SEAttention(planes)
        elif(self.ATTENTION_TYPE == 'ELA'):
            self.attention = EfficientLocalizationAttention(planes, kernel_size=self.KERNEL_SIZE, num_groups=self.NUMBER_GROUPS , group_setting=self.GROUP_SETTING)
        elif(self.ATTENTION_TYPE == "CA") :
            self.attention = CoordAtt(planes, planes//2, planes//2)
        elif(self.ATTENTION_TYPE == "ECA") :
            self.attention = ECAAttention(planes, kernel_size=self.KERNEL_SIZE)
        elif(self.ATTENTION_TYPE == "BAM") :
            self.attention = BAMBlock(planes, planes//2, planes//2)
        elif(self.ATTENTION_TYPE == "CBAM") :
            self.attention = CBAMBlock(planes, planes//2, planes//2)        
        elif(self.ATTENTION_TYPE == "A2") :
            self.attention = DoubleAttention(planes, planes//2, planes//2)                
        else:
            self.attention = nn.Identity()
        

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
        out = self.attention(out)
        
        out += identity
        out = self.relu(out)


        return out
def get_resnet18(attention_type , ela_kernelsize , ela_group_setting , ela_numgroup , **kwargs):
    """
    Creates a ResNet-18 model with ELA blocks.
    """
    model = None
    if attention_type is not None:
        
        BasicBlockWithAttention.ATTENTION_TYPE = attention_type
        BasicBlockWithAttention.KERNEL_SIZE = ela_kernelsize
        BasicBlockWithAttention.NUMBER_GROUPS = ela_numgroup
        BasicBlockWithAttention.GROUP_SETTING = ela_group_setting
        
        model = ResNet(BasicBlockWithAttention, [2, 2, 2, 2], **kwargs)
    else :
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model    
