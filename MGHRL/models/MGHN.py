import torch
import torch.nn as nn
from dhg.nn import UniGATConv, MultiHeadWrapper
from models.modules import AttBlock, DimReschedule, LayerNorm
from timm.models.layers import trunc_normal_


class Fusion(nn.Module):
    def __init__(self, level, dims, is_first) -> None:
        super().__init__()
        
        self.level = level
        self.is_first = is_first
        self.down = nn.Sequential(
                DimReschedule(dims[level-1], dims[level]),
                LayerNorm(dims[level], eps=1e-6),
            ) if level in [1, 2, 3] else nn.Identity()
        self.up = DimReschedule(dims[level + 1], dims[level]) if not is_first and level in [0, 1, 2] else nn.Identity() 
        self.gamma = nn.Parameter(torch.ones(1)) if level in [1, 2] and not is_first else None

    def forward(self, x_down, x_up=None):
        if self.is_first:
            return self.down(x_down)

        if self.level == 3:
            return self.down(x_down)
        else:
            x_down_rescaled = self.down(x_down)
            x_up_rescaled = self.up(x_up)
            if self.gamma is not None:
                return x_up_rescaled + self.gamma * x_down_rescaled
            else:
                return x_up_rescaled + x_down_rescaled
            
class Level(nn.Module):
    def __init__(self, level, dims, layers, is_first, dp_rate=0.0) -> None:
        super().__init__()
        countlayer = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, dims, is_first)
        modules = [AttBlock(dims[level], expansion*dims[level], dims[level], num_heads=4, layer_scale_init_value=1e-6, drop_path=dp_rate[countlayer+i]) for i in range(layers[level])]
        self.blocks = nn.Sequential(*modules)
    def forward(self, x_down, x_up, hg):
        x = self.fusion(x_down, x_up)
        for module in self.blocks:
            x = module(x, hg)
        return x

class SubNet(nn.Module):
    def __init__(self, dims, layers, is_first, dp_rates, save_memory) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, dims[0])), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, dims[1])), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, dims[2])), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, dims[3])), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 

        self.level0 = Level(0, dims, layers, is_first, dp_rates)

        self.level1 = Level(1, dims, layers, is_first, dp_rates)

        self.level2 = Level(2, dims, layers, is_first, dp_rates)

        self.level3 = Level(3, dims, layers, is_first, dp_rates)

    def forward(self, *args):
        
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        
        x, hg, output_l0, output_l1, output_l2, output_l3= args

        output_l0 = (self.alpha0)*output_l0 + self.level0(x, output_l1, hg)
        output_l1 = (self.alpha1)*output_l1 + self.level1(output_l0, output_l2, hg)
        output_l2 = (self.alpha2)*output_l2 + self.level2(output_l1, output_l3, hg)
        output_l3 = (self.alpha3)*output_l3 + self.level3(output_l2, None, hg)
        return output_l0, output_l1, output_l2, output_l3

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign
            
class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()        

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_channels, eps=1e-6),  # 对节点特征进行层归一化
            nn.Linear(in_channels, num_classes),  # 通过全连接层映射到分类数目
        )
    def forward(self, x):
        return self.classifier(x)
            
class MGHN(nn.Module):
    def __init__(
            self,
            num_subnet: int,
            in_channels: int,
            num_classes: int,
            num_heads: int = 4,
            dims=[32, 64, 96, 128],
            layers=[2, 3, 6, 3],
            drop_path=0.0,
            save_memory=False,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            atten_neg_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_subnet = num_subnet
        self.dims = dims
        self.layers = layers

        self.in_layer_conv = UniGATConv(
                                in_channels,
                                dims[0],
                                use_bn=use_bn,
                                drop_rate=drop_rate,
                                atten_neg_slope=atten_neg_slope,
                                is_last=False,
                            )
        self.in_layer_norm = LayerNorm(dims[0], eps=1e-6)

        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]

        self.out_layer = UniGATConv(
            dims[-1],
            num_classes,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=True,
        )

        for i in range(num_subnet):
            is_first = True if i == 0 else False
            self.add_module(
                f'subnet{str(i)}',
                SubNet(dims, layers, is_first, dp_rates=dp_rate, save_memory=save_memory)
            )

        self.apply(self._init_weights)

    def forward(self, x, hg):
        output_l0, output_l1, output_l2, output_l3 = 0, 0, 0, 0
        x = self.in_layer_conv(x, hg)
        x = self.in_layer_norm(x)

        for i in range(self.num_subnet):
            output_l0, output_l1, output_l2, output_l3 = getattr(
                self, f'subnet{str(i)}'
            )(x, hg, output_l0, output_l1, output_l2, output_l3)
        return self.out_layer(output_l3, hg)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):  
            trunc_normal_(module.weight, std=.02)  
            if module.bias is not None:  
                nn.init.constant_(module.bias, 0) 

