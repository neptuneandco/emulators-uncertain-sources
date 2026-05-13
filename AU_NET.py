import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class DoubleConv3D(nn.Module):
    """(Convolution3D -> GroupNorm -> GELU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            #nn.GroupNorm(num_groups=8, num_channels=mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is the feature map from the previous layer in the decoder
        # x2 is the skip connection from the corresponding layer in the encoder
        x1 = self.up(x1)
        
        # Handle potential padding differences between x1 and x2 from skip connection
        # Input format is (N, C, D, H, W)
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        # Pad x1 to match the dimensions of x2
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionBlock(nn.Module):
    """Self-attention block for the U-Net bottleneck"""
    def __init__(self, in_channels, n_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads

        # Normalization layer before attention
        self.norm = nn.LayerNorm(in_channels)
        # Multi-head self-attention module
        self.attention = nn.MultiheadAttention(in_channels, n_heads, batch_first=True)
        # A simple feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, 4 * in_channels),
            nn.GELU(),
            nn.Linear(4 * in_channels, in_channels),
        )

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, channels, time, height, width)
        """
        b, c, t, h, w = x.shape
        
        # Reshape and permute for attention mechanism
        x_reshaped = x.view(b, c, -1).permute(0, 2, 1) # (b, t*h*w, c)
        
        x_norm = self.norm(x_reshaped)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x_reshaped = attn_output + x_reshaped # Residual connection
        ffn_output = self.ffn(x_reshaped)
        x_reshaped = ffn_output + x_reshaped # Residual connection
        
        out = x_reshaped.permute(0, 2, 1).view(b, c, t, h, w)
        
        return out

class UNet3DAttention(nn.Module):
    def __init__(self, in_channels, out_channels, base_c=16):
        super(UNet3DAttention, self).__init__()
        # This architecture is specifically designed for two outputs (heads and concentrations)
        assert out_channels == 2, "This split-head architecture is designed for 2 output channels."
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv3D(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        
        self.bottleneck_attn = AttentionBlock(in_channels=base_c * 8, n_heads=8)

        self.up1 = Up(base_c * 8, base_c * 4)
        self.up2 = Up(base_c * 4, base_c * 2)
        self.up3 = Up(base_c * 2, base_c)

        # A separate final convolution for each task. Each outputs 1 channel.
        self.out_head_heads = nn.Conv3d(base_c, 1, kernel_size=1)
        self.out_head_conc = nn.Conv3d(base_c, 1, kernel_size=1)

    def forward(self, x):
        # Input shape: (batch, height, width, time, channels)
        x = x.permute(0, 4, 3, 1, 2)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_attn = self.bottleneck_attn(x4)

        x = self.up1(x_attn, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        heads_logits = self.out_head_heads(x) # Shape: (batch, 1, time, height, width)
        conc_logits = self.out_head_conc(x)   # Shape: (batch, 1, time, height, width)

        # Concatenate the results along the channel dimension
        logits = torch.cat([heads_logits, conc_logits], dim=1) # Shape: (batch, 2, time, height, width)

        # (batch, height, width, time, out_channels)
        output = logits.permute(0, 3, 4, 2, 1)

        return output

if __name__ == '__main__':

    batch_size = 2
    input_channels = 6 
    time_depth = 16   
    height = 64       
    width = 64     
    
    output_channels = 2
    
    model = UNet3DAttention(
        in_channels=input_channels, 
        out_channels=output_channels,
        base_c=32 
    )
    

    dummy_input = torch.randn(batch_size, height, width, time_depth, input_channels)
    
    print(f"Input shape (channels-last): {dummy_input.shape}")
    predicted_output = model(dummy_input)
    print(f"Predicted Output shape (channels-last): {predicted_output.shape}")
    expected_shape = (batch_size, height, width, time_depth, output_channels)
    assert predicted_output.shape == expected_shape, "Output shape is incorrect!"   
    print("\nModel with split heads created and test forward pass successful!")
