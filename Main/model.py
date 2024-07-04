import torch
from torch import nn
from torchvision.models import vgg16, mobilenet_v3_large, resnet50
from torchvision.models import VGG16_Weights, MobileNet_V3_Large_Weights, ResNet50_Weights

class VGG16Binary(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16Binary, self).__init__()
        if pretrained:
            self.vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            self.vgg16 = vgg16(weights=None)
        self.dropout = nn.Dropout(p=0.5)
        self.vgg16.classifier[6] = nn.Linear(self.vgg16.classifier[6].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vgg16(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        return x

class MobileNetV3Binary(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3Binary, self).__init__()
        if pretrained:
            self.mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        else:
            self.mobilenet = mobilenet_v3_large(weights=None)
        self.dropout = nn.Dropout(p=0.5)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        return x

class ResNet50Binary(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Binary, self).__init__()
        if pretrained:
            self.resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet50 = resnet50(weights=None)
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.resnet50.fc.in_features, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet50(x)
        x = self.sigmoid(x)
        return x

def get_model(model_name, pretrained=True):
    if model_name == 'VGG16Binary':
        return VGG16Binary(pretrained=pretrained)
    elif model_name == 'MobileNetV3Binary':
        return MobileNetV3Binary(pretrained=pretrained)
    elif model_name == 'ResNet50Binary':
        return ResNet50Binary(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
