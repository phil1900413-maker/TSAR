import math
import torch
import torch.nn as nn
import timm
from timm.models._efficientnet_blocks import SqueezeExcite


class _SqueezeExcite1D(nn.Module):
    def __init__(self, se: SqueezeExcite):
        super().__init__()
        self.conv_reduce = se.conv_reduce
        self.act1 = se.act1
        self.conv_expand = se.conv_expand
        self.gate = se.gate
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        s = self.pool(x)
        s = self.conv_reduce(s)
        s = self.act1(s)
        s = self.conv_expand(s)
        return x * self.gate(s)


def _replace_2d_by_1d(m: nn.Module):
    for name, child in list(m.named_children()):
        if len(list(child.children())):
            _replace_2d_by_1d(child)

        if isinstance(child, SqueezeExcite):
            setattr(m, name, _SqueezeExcite1D(child))
            continue

        if isinstance(child, nn.Conv2d):
            setattr(
                m,
                name,
                nn.Conv1d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size[0],
                    stride=child.stride[0],
                    padding=child.padding[0],
                    dilation=child.dilation[0],
                    groups=child.groups,
                    bias=(child.bias is not None),
                ),
            )
            continue

        if isinstance(child, nn.BatchNorm2d):
            setattr(
                m,
                name,
                nn.BatchNorm1d(
                    num_features=child.num_features,
                    eps=child.eps,
                    momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats,
                ),
            )
            continue

        if isinstance(child, nn.MaxPool2d):
            setattr(
                m,
                name,
                nn.MaxPool1d(
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    ceil_mode=child.ceil_mode,
                ),
            )
            continue

        if isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(m, name, nn.AdaptiveAvgPool1d(output_size=child.output_size))
            continue

        if isinstance(child, timm.layers.SelectAdaptivePool2d):
            setattr(m, name, nn.AdaptiveAvgPool1d(output_size=1))
            continue


def create_efficientnet_1d(name: str, in_chans: int = 1):
    net = timm.create_model(name, pretrained=True, in_chans=in_chans, num_classes=1)
    _replace_2d_by_1d(net)
    if hasattr(net, "classifier"):
        net.classifier = nn.Identity()
    if hasattr(net, "global_pool"):
        net.global_pool = nn.Identity()
    return net


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 6000):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.drop(x + self.pe[: x.size(1)])


class HybridEffNetB5Transformer(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        input_points: int = 2203,
        backbone_name: str = "efficientnet_b5",
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        mu_bias_init: float = -4.0,
        log_sigma_bias_init: float = math.log(0.5),
    ):
        super().__init__()

        self.cnn_backbone = create_efficientnet_1d(backbone_name, in_chans=1)

        with torch.no_grad():
            dummy = torch.randn(1, 1, input_points)
            feat = self.cnn_backbone(dummy)
            c_out = feat.shape[1]

        self.input_proj = nn.Conv1d(c_out, d_model, kernel_size=1)
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.mu_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_outputs))
        self.sigma_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_outputs))

        self._mu_bias_init = float(mu_bias_init)
        self._log_sigma_bias_init = float(log_sigma_bias_init)
        self.apply(self._init_linear)

    def _init_linear(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
                if m is self.mu_head[-1]:
                    m.bias.data.fill_(self._mu_bias_init)
                if m is self.sigma_head[-1]:
                    m.bias.data.fill_(self._log_sigma_bias_init)

    def get_param_groups(self, backbone_lr: float, head_lr: float):
        backbone = {"params": self.cnn_backbone.parameters(), "lr": float(backbone_lr)}
        head_params = [p for n, p in self.named_parameters() if not n.startswith("cnn_backbone.")]
        head = {"params": head_params, "lr": float(head_lr)}
        return [backbone, head]

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        f = self.cnn_backbone(x)
        f = self.input_proj(f)
        f = f.permute(0, 2, 1)

        f = self.pos(f)
        z = self.encoder(f)
        z = z.mean(dim=1)

        mu = self.mu_head(z)
        log_sigma = self.sigma_head(z)
        return torch.cat([mu, log_sigma], dim=1)
