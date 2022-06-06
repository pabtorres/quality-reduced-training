import torchvision.models as models
from dataset_classes_vit import *
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
from thesis_persist_as_list import persist_list
import os

import csv

# Variables para red neuronal
BATCH_SIZE = 32
DEVICE = 'cuda'


# Transform (also check anti_transform in mepis_utils accordingly)
transform = torchvision.transforms.Compose(
      [torchvision.transforms.Resize(256),
      torchvision.transforms.CenterCrop(224),
      torchvision.transforms.ToTensor(),])

# Load Validation Dataset (No Shuffle)
test_loader_original = DataLoader(TrainingDataset(transform=transform, list='humanet_minival_set_v3_f.json'), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_color = DataLoader(ValidationDataset(transform=transform, list='testing_human_color.json'), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_combined = DataLoader(ValidationDataset(transform=transform, list='testing_human_combined.json'), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_crop = DataLoader(ValidationDataset(transform=transform, list='testing_human_crop.json'), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_resolution = DataLoader(ValidationDataset(transform=transform, list='testing_human_resolution.json'), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Lista de dataloaders
data_loader_list = [test_loader_original, test_color, test_combined, test_crop, test_resolution]
names_loader_list = ['Original', 'Color', 'Combined', 'Crop', 'Resolution']

def results_per_class(net, folder):
  # Neural Network to evaluation mode
  net.eval()

  # Nueva lista de accuracys por cada CHECKPOINT
  lista_de_accuracys = []

  # Testear sobre conjuntos
  for test_loader, name_loader in zip(data_loader_list, names_loader_list):
    running_acc = 0.0
    total_test = len(test_loader.dataset)

    # CSV GEN
    csv_file = open(f'{folder}/{name_loader}_results.csv', 'w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['Gold', 'Predicted', 'CorrectORIncorrect'])

    # Iterate
    with torch.no_grad():
      for i, data in enumerate(test_loader):
        X, Y = data
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        Y_pred = net(X)['logits']
        _, idx_pred = torch.max(Y_pred, dim=1)
        is_equal = (idx_pred == Y)
        for elm, predo, elo in zip(Y, idx_pred, is_equal):
          csv_writer.writerow([int(elm), int(predo), int(elo)])
        running_acc += torch.sum(idx_pred == Y).item()
        avg_acc = running_acc/total_test*100
      #print(f'{name_loader}: {avg_acc}')
      lista_de_accuracys.append(avg_acc)

  return lista_de_accuracys