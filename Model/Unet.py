import torch
import torch.nn as nn
from Model.Downblock import DownBlock
from Model.Midblock import MidBlock
from Model.Upblock import Upblock
from Positional_Embedding.positional_embedding import time_embedding


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        im_channels = 3
        self.down_channels = [32, 64, 128, 256]
        self.mid_channels = [256, 256, 128]
        self.t_emb_dim = 128
        self.down_sample = [True, True, False]
        self.num_heads = 4

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1

        # Time embedding projection
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)

        # Downsampling layers
        self.downs = nn.ModuleList([
            DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim, down_sample=self.down_sample[i],num_heads=self.num_heads)
            for i in range(len(self.down_channels) - 1)
        ])

        # Mid layers
        self.mids = nn.ModuleList([
            MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,num_heads=self.num_heads)
            for i in range(len(self.mid_channels) - 1)
        ])

        # Upsampling layers
        self.ups = nn.ModuleList([
            Upblock(self.down_channels[i]*2, self.down_channels[i-1] if i != 0 else 16, self.t_emb_dim, up_sample=self.down_sample[i])
            for i in reversed(range(len(self.down_channels) - 1))
        ])

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        out = self.conv_in(x)  # B x C1 x H x W

        # Time embedding
        t_emb = time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)

        for mid in self.mids:
            out = mid(out, t_emb)

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)

        return out
