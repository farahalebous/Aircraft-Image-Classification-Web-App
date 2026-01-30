import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(hidden_size, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


def resnet50(num_classes, device='cpu'):
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    for param in resnet.parameters():
        param.requires_grad = False
    
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device)
    
    return resnet


def alexnet(num_classes, device='cpu'):
    alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    
    for param in alexnet.parameters():
        param.requires_grad = False
    
    alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, num_classes)
    alexnet = alexnet.to(device)
    
    return alexnet

