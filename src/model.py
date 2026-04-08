import torch
import torch.nn as nn
import torch.nn.functional as F

# Used in Vanilla UNet
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # here we choose padding k=3 and p=1 so that the size of the image doesn't change
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # inplace=True will reduce the memory usage
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# For residual connections
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # here we choose padding k=3 and p=1 so that the size of the image doesn't change
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # inplace=True will reduce the memory usage
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        # 1x1 projection when channel dimensions differ
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip=True, residual=False):
        super().__init__()
        self.skip = skip
        # Hout​=(Hin​−1)⋅stride−2⋅padding+kernel_size+output_padding
        # H_out = (H_in - 1)*2 - 2*0 + 2 + 0 = 2 * H_in
        # Hout = 2*H_in ( we get the correct up sampling )
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        
        ConvBlock = ResidualBlock if residual else DoubleConv
        if skip:
            self.conv = ConvBlock(in_ch, out_ch)
        else:
            self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)

        if self.skip:
            # handle size mismatch (important!)
            # if x.shape != skip.shape:
            #     x = F.interpolate(x, size=skip.shape[2:])
            # instead of handeling size mismatch we throw an error. ( so that correct concat occurs s)
            assert skip is not None, "skip tensor required when skip=True"
            assert x.shape[2:] == skip.shape[2:], f"Shape mismatch: {x.shape} vs {skip.shape}"
            
            x = torch.cat([skip, x], dim=1)  # concat channels
        
        return self.conv(x)
    
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, residual=False):
        super().__init__()
        ConvBlock = ResidualBlock if residual else DoubleConv
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_down = self.pool(x_conv)
        return x_conv, x_down   # return skip + downsampled

class Mycool_UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=13, skip=True, residual=False):
    #   here num_classes is 13 as there are 13 classes in the dataset
        super().__init__()
        self.skip = skip

        # Encoder
        self.down1 = Down(in_ch=in_ch, out_ch=64, residual=residual)     
        self.down2 = Down(in_ch=64, out_ch=128, residual=residual)             
        self.down3 = Down(in_ch=128, out_ch=256, residual=residual)            
        self.down4 = Down(in_ch=256, out_ch=512, residual=residual)            

        # Bottleneck
        ConvBlock = ResidualBlock if residual else DoubleConv
        self.bottleneck = ConvBlock(in_ch=512, out_ch=1024)

        # Decoder
        self.up1 = Up(in_ch=1024, out_ch=512, skip=skip, residual=residual)               
        self.up2 = Up(in_ch=512, out_ch=256, skip=skip, residual=residual)                
        self.up3 = Up(in_ch=256, out_ch=128, skip=skip, residual=residual)                
        self.up4 = Up(in_ch=128, out_ch=64, skip=skip, residual=residual)                 

        # Standard Final layer
        # self.final_conv = nn.Conv2d(in_ch=64, out_ch=out_ch, kernel_size=1)
        
        # Modified UNet as per our needs
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        if self.skip:
            x = self.up1(x, skip=s4)
            x = self.up2(x, skip=s3)
            x = self.up3(x, skip=s2)
            x = self.up4(x, skip=s1)
        else:
            x = self.up1(x)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)

        # Output for vanilla Unet ( single output )
        # return self.final_conv(x)

        # Output for modified Unet ( two outputs )
        seg = self.seg_head(x)
        depth = self.depth_head(x)

        return seg, depth