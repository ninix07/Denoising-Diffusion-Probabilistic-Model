import torch
import torch.nn as nn 


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()

        self.first_layer = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
        ])
        
        self.second_layer = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),  # Fixed here
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),  # Fixed here
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        
        self.attention_norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)  # Fixed here
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Fixed input channels here
        ])
        
    def forward(self, X, t_emb):
        out = X
        out = self.first_layer[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.second_layer[0](out)
        out = out + self.residual_input_conv[0](X)

        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norm(in_attn)  # Fixed here
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        resnet_input = out
        out = self.first_layer[1](out)
        out = out + self.t_emb_layers[1](t_emb)[:, :, None, None]
        out = self.second_layer[1](out)
        out = out + self.residual_input_conv[1](resnet_input)
        
        return out
