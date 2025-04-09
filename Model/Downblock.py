import torch
import torch.nn as nn 


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample= down_sample
        self.first_layer= nn.Sequential(
            nn.GroupNorm(num_groups=8,num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size=3, stride=1, padding=1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim,out_channels)
        )
        self.second_layer= nn.Sequential(
            nn.GroupNorm(num_groups=8,num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1)
        )
        self.attention_norm =  nn.GroupNorm(num_groups=8,num_channels=out_channels)
        self.attention= nn.MultiheadAttention(out_channels,num_heads,batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.down_sample_layer= nn.Conv2d(out_channels,out_channels,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1) if self.down_sample else nn.Identity()
        
    def forward(self,x,t_emb):
            out=x 
            out = self.first_layer(out)
            out = out + self. t_emb_layers(t_emb)[:,:,None, None]
            out= self.second_layer(out)
            out= out +self.residual_input_conv(x)
            #Attention block
            batch_size,channels, h,w = out.shape
            in_attn= out.reshape(batch_size, channels, h*w)
            in_attn= self.attention_norm(in_attn)
            in_attn= in_attn.transpose(1,2)
            out_attn,_ = self.attention(in_attn,in_attn,in_attn)
            out_attn=out_attn.transpose(1,2).reshape(batch_size,channels, h,w)
            out = out+out_attn

            out= self.down_sample_layer(out)
            return out
