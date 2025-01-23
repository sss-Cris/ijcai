import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from dhg.nn import UniGATConv, MultiHeadWrapper


class DimReschedule(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.channel_reschedule = nn.Sequential(
                                        nn.Linear(inchannel, outchannel),
                                        LayerNorm(outchannel, eps=1e-6),
                                        nn.ReLU())
    def forward(self, x):
        x = self.channel_reschedule(x)
        return x
    
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine = True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    
    
class AttBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_heads: int = 4,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
        layer_scale_init_value=1e-6,
        drop_path= 0.0,
    ) -> None:
        super(AttBlock, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.drop_layer = nn.Dropout(drop_rate)
        self.multi_head_layer = MultiHeadWrapper(
            num_heads,
            "concat",
            UniGATConv,
            in_channels=in_channels,
            out_channels=hid_channels//num_heads,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
        )
        self.norm = nn.LayerNorm(hid_channels, eps=1e-6)
        self.act = nn.GELU()
        self.conv = nn.Linear(hid_channels, out_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        input = X
        X = self.drop_layer(X)
        X = self.multi_head_layer(X=X, hg=hg)
        X = self.norm(X)
        X = self.act(X)
        X = self.conv(X)
        if self.gamma is not None:
            X = self.gamma * X
        X = input + self.drop_path(X)
        return X