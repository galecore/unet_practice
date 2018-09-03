from scipy.misc import toimage
import torchvision, torch, numpy
from PIL import Image
import os

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

transforms = torchvision.transforms.Compose(
[
    torchvision.transforms.Resize((320, 320)), 
    torchvision.transforms.ToTensor()     
])

# path_to_weights = "state_dict.wght"


def load_net():
    net = UNet()
    net.load_state_dict(torch.load("state_dict.wght"))
    print("loaded")
    net.eval()
    return net

def prediction_to_image(prediction):
    prediction = prediction.squeeze(0).squeeze(0).numpy()
    return toimage(prediction)

def save_img(img, path):
    img.save(path)

def get_ext(path):
    return "." + path.split(".")[-1]

def extend_filename(path, info):
    ext = get_ext(path)
    return path.replace(ext, "_" + info + ext)

def get_filename(path):
    return path.split("/")[-1]

def make_prediction(img):
    net = load_net()
    prediction = net.forward(img.unsqueeze(0))
    pred_img = prediction_to_image(prediction.data)
    return pred_img

def process_photo(filepath):
    print(filepath)
    transformed_filepath = extend_filename(filepath, "transformed")
    predicted_filepath = extend_filename(filepath, "pred")
    emphed_filepath = extend_filename(filepath, "emph")

    img = transforms(Image.open(filepath))
    pred_img = make_prediction(img)

    save_img(toimage(img), transformed_filepath)
    save_img(pred_img, predicted_filepath)
    save_img(toimage(img*transforms(pred_img)), emphed_filepath)

    return get_filename(transformed_filepath), get_filename(predicted_filepath), get_filename(emphed_filepath)
