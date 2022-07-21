import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

import attacks.iugm_attack as attackU
import attacks.itfgsm_attack as attackT
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channels = 3

class CNN(nn.Module):
  def __init__(self, ):
    super().__init__()
    
    self.conv1 = nn.Conv2d(in_channels = channels, out_channels=16, kernel_size = (3,3))
    self.conv2 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size = (3,3))
    self.conv3 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = (3,3))
    self.conv4 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size = (3,3))
    
    self.batchnorm1 = nn.BatchNorm2d(32)
    self.batchnorm2 = nn.BatchNorm2d(128)
    self.batchnorm3 = nn.BatchNorm1d(512)

    self.maxpool1 = nn.MaxPool2d((2,2))
    self.maxpool2 = nn.MaxPool2d((2,1))
    self.dropout = nn.Dropout(0.5)

    self.linear1 = nn.Linear(123904,512)
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
  model = CNN()
  model.load_state_dict(torch.load(MODEL_CHECKPOINT_FILE, map_location=device))
  # model = torch.load(MODEL_CHECKPOINT_FILE)
  model.to(device)
  model.eval()

  return model

def prediction(path, m):
  image = torchvision.io.read_image(path)
  image = image.unsqueeze(0)
  image = F.interpolate(image, size=100)
  image = image/255
  image = image.float()
  y_pred = m(image)
  y_pred = F.softmax(y_pred, dim=1)
  conf, y_pred_in = torch.max(y_pred, 1)

  return conf.detach().numpy()[0] * 100, y_pred_in.numpy()[0], image

def attack(filename, image, m, original_label, att):

  generated_name = "g_" + filename + "_" + att + ".jpg"
  noise_name = "n_" + filename + "_" + att + ".jpg"

  generated_name = generated_name.replace("/", "")
  noise_name = noise_name.replace("/", "")

  # need grad for attack to work
  image_grad = image.clone().detach().requires_grad_(True)

  # figure out which kind of attack first
  att_cat = att.split(" - ")[0]
  isSuccess = False

  if att_cat == "Untargetted":
    # untargetted attack
    new_image = attackU.iugm_attack(image_grad, attOptions[att], m, original_label)
    isSuccess = True
  elif att_cat == "Targetted":
    new_image, isSuccess = attackT.itfgsm_attack(image_grad, 0.005, m, original_label, attOptions[att])
    print('Is targetted attack successful? ', isSuccess)

  if isSuccess:
    img = torchvision.transforms.ToPILImage()(new_image)
    img.save(os.path.join(GENERATED_FOLDER, generated_name))

    # generate noise
    image = image.squeeze(0)
    noise = torch.subtract(image, new_image)
    img = torchvision.transforms.ToPILImage()(noise)
    img.save(os.path.join(NOISE_FOLDER, noise_name))

    # get new iamge prediction
    new_image = new_image.unsqueeze(0)
    y_pred = m(new_image)
    y_pred = F.softmax(y_pred, dim=1)
    conf, y_pred_in = torch.max(y_pred, 1)

    return conf.detach().numpy()[0] * 100, y_pred_in.numpy()[0], generated_name, noise_name
  else:
    return 0, 0, "", ""