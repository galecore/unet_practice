import torch
import numpy

class UNetConvBlock(torch.nn.Module):
    def __init__(self, in_layers, out_layers, kernel_size=7, padding=3, activation=torch.nn.ReLU, pooling=True):
        super(UNetConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_layers, out_layers, kernel_size, padding=padding)
        self.bnorm1 = torch.nn.BatchNorm2d(out_layers)
        self.conv2 = torch.nn.Conv2d(out_layers, out_layers, kernel_size, padding=padding)
        self.bnorm2 = torch.nn.BatchNorm2d(out_layers)
        self.pooling = pooling
        self.pool = torch.nn.MaxPool2d(2)
        self.activation = activation()
        
    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.bnorm1(out)
        out = self.activation(self.conv2(out))
        out = self.bnorm2(out)
        if (self.pooling):
            out = self.pool(out)
        return out     
    
class UNetUpConvBlock(torch.nn.Module):
    def __init__(self, in_layers, out_layers, kernel_size=3, padding=1, activation=torch.nn.ReLU):
        super(UNetUpConvBlock, self).__init__()
        self.upconv = torch.nn.Conv2d(in_layers, 4*in_layers, 1)
        self.pixelshuffle = torch.nn.PixelShuffle(2)
        self.conv = UNetConvBlock(in_layers, out_layers, pooling=False)  
        
    def forward(self, x):
        out = self.upconv(x)
        out = self.pixelshuffle(out)
        out = self.conv(out)
        return out    

def stack(old, new):
    return torch.cat([old, new], dim=1)

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.conv1 = UNetConvBlock(3, 16)
        self.conv2 = UNetConvBlock(16, 32)
        self.conv3 = UNetConvBlock(32, 64)

        self.upconv3 = UNetUpConvBlock(64, 32) # 64 -> 32
        self.upconv2 = UNetUpConvBlock(64, 16)  # stack with conv2 32 + 32 -> 16
        self.upconv1 = UNetUpConvBlock(32, 8) # stack with conv1 16 + 16 -> 8
        
        self.fullconv = torch.nn.Conv2d(11, 1, 1) # with initial 8 + 3 -> 1 
        self.pred = torch.nn.Sigmoid()
        
    def forward(self, x):
        initial = x.clone()
        
        c1 = self.conv1(x)
        c2 = self.conv2(c1)          
        x = self.conv3(c2)
        
        x = self.upconv3(x)
        
        x = stack(c2, x)
        x = self.upconv2(x)
        
        x = stack(c1, x)
        x = self.upconv1(x)
        
        x = stack(initial, x)
        x = self.fullconv(x)  
        x = self.pred(x)
        return x