import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision

data_dir = os.path.join('.', 'data', 'gtsrb')
train_path = os.path.join('.', 'data', "gtsrb", 'Train')
test_path = os.path.join('.', 'data', 'gtsrb')

IMG_HEIGHT = 100
IMG_WIDTH = 100
channels = 3

NUM_CATEGORIES = len(os.listdir(train_path))

class ImageDataset(Dataset):
    def __init__(self, device, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = T.Compose([T.Resize((IMG_HEIGHT,IMG_WIDTH))])

        self.image_paths = [] 
        self.image_labels = []
        self.get_image_paths()
        self.device = device

    def get_image_paths(self):
      for i in range(NUM_CATEGORIES):
        path = os.path.join(train_path, str(i))
        images = os.listdir(path)
        for img in images:
          self.image_paths.append(path + '/' + img)
          self.image_labels.append(i)

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.image_paths[idx])
        resize_image = self.transform(image)
        # image_fromarray = Image.fromarray(image, 'RGB')
        # resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
        divided_image = resize_image/255
        img_tensor = torch.tensor(divided_image).float().to(self.device)

        label = F.one_hot(torch.tensor(self.image_labels[idx]), NUM_CATEGORIES)
        label.to(self.device)

        if self.transform:
            sample = self.transform(image)

        return img_tensor, label
