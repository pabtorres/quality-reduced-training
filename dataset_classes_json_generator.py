from torchvision.transforms import ToTensor
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import *
import torch
from torchvision import datasets, transforms as T
import json

img_to_tensor = ToTensor()
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

class GenDataset(Dataset):
  def __init__(self, transform, list='training_data_list_1percent.json'):
    self.transform = transform
    with open(list, 'r') as f:
      self.list = json.load(f)

  def __getitem__(self, index):
    img_dir, label = self.list[index]
    img = Image.open(img_dir)
    if len(img.getbands()) != 3: img = img.convert("RGB")
    tensor = self.transform(img)
    return tensor, torch.tensor(int(label)), img_dir

  def __len__(self):
    return len(self.list)

class MEPIDataset(Dataset):
  def __init__(self, transform, list='training_data_list_1percent.json', mepi_list='mepi_test/json_file.json'):
    self.transform = transform
    with open(list, 'r') as f:
      self.list = json.load(f)
    with open(mepi_list, 'r') as g:
      self.mepis = json.load(g)

  def __getitem__(self, index):
    img_dir, label = self.list[index]
    img = Image.open(img_dir)
    if len(img.getbands()) != 3: img = img.convert("RGB")
    tensor = self.transform(img)
    return tensor, torch.tensor(int(label)), str(self.mepis[img_dir])

  def __len__(self):
    return len(self.list)

class TrainingDataset(Dataset):
  def __init__(self, transform, list='training_data_list_1percent.json'):
    self.transform = transform
    with open(list, 'r') as f:
      self.list = json.load(f)

  def __getitem__(self, index):
    img_dir, label = self.list[index]
    img = Image.open(img_dir)
    if len(img.getbands()) != 3: img = img.convert("RGB")
    tensor = self.transform(img)
    return tensor, torch.tensor(int(label))

  def __len__(self):
    return len(self.list)

class ValidationDataset(Dataset):
  def __init__(self, transform, list='validation_data_list.json'):
    self.transform = transform
    with open(list, 'r') as f:
      self.list = json.load(f)

  def __getitem__(self, index):
    img_dir, label = self.list[index]
    img = Image.open(img_dir)
    if len(img.getbands()) != 3: img = img.convert("RGB")
    tensor = self.transform(img)
    return tensor, torch.tensor(int(label))

  def __len__(self):
    return len(self.list)