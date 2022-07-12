import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import *

channels = 3

class CNN(nn.Module):
  def __init__(self, ):
    super().__init__()
    
    self.conv1 = nn.Conv2d(in_channels = channels, out_channels=16, kernel_size = (3,1))
    self.conv2 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size = (3,1))
    self.conv3 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = (3,1))
    self.conv4 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size = (3,1))
    
    self.batchnorm1 = nn.BatchNorm2d(32)
    self.batchnorm2 = nn.BatchNorm2d(128)
    self.batchnorm3 = nn.BatchNorm1d(512)

    self.maxpool1 = nn.MaxPool2d((2,2))
    self.maxpool2 = nn.MaxPool2d((2,1))
    self.dropout = nn.Dropout(0.5)

    self.linear1 = nn.Linear(140800,512)
    # self.linear1 = nn.Linear(512,512)
    self.linear2 = nn.Linear(512,43)

  def forward(self, x):

    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = self.maxpool1(out)
    out = self.batchnorm1(out)
    
    out = F.relu(self.conv3(out))
    out = F.relu(self.conv4(out))
    out = self.maxpool2(out)
    out = self.batchnorm2(out)

    out = torch.flatten(out,1)
    out = self.linear1(out)
    out = self.batchnorm3(out)
    out = self.dropout(out)

    out = self.linear2(out)

    return out

def loadModel():

  device = torch.device('cpu')
  model = CNN()
  model.load_state_dict(torch.load("./models/model_bs1024_100_sd.pt", map_location=device))
  model.eval()

  return model

def prediction(path, m):
  image = torchvision.io.read_image(path)
  image = image.unsqueeze(0)
  image = F.interpolate(image, size=100)
  image = image/255
  image = image.float()
  y_pred = m(image)
  _, y_pred_in = torch.max(y_pred, 1)

  return classes[y_pred_in.numpy()[0]]
