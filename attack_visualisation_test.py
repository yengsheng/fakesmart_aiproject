""" Gradient Attacks
## Iterative Untargeted Gradient Attack
## iterative targeted fast gradient sign attack
## Carlini-Wagner Attack
## Visualisations
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attacks import attack_common, itfgsm_attack, iugm_attack, carlini_wagner_attack

from custom_dataset import ImageDataset, NUM_CATEGORIES
from model import loadModel, CNN
import datetime

np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_dataset = ImageDataset(device)
# train_size = int(0.75 * len(full_dataset)) # basic attacks
train_size = int(0.992 * len(full_dataset)) # CW attacks
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
print("Tested on ", test_size, "images")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = loadModel()


print("GPU Detected:", torch.cuda.is_available())
print("start:", datetime.datetime.now())

# attack_common.test_and_visualise_basic_attack(model, iugm_attack.test_iugm_attack, test_dataloader, test_size, "iugm_attack")
# attack_common.test_and_visualise_basic_attack(model, itfgsm_attack.test_itfgsm_attack, test_dataloader, test_size, "itfgsm_attack")
carlini_wagner_attack.test_and_visualise_cw_attack(model, test_dataloader, test_size, NUM_CATEGORIES, 1)
print("end:", datetime.datetime.now())
