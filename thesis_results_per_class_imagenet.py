import torchvision.models as models
#from dataset_classes_vit import *
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
from thesis_persist_as_list import persist_list
import os
import sys
import torchvision.transforms as transforms

from dataloaders_for_imagenet import *

# Redes neuronales
from thesis_networks_controller_imagenet import instantiate_network

import csv

# Variables para red neuronal
BATCH_SIZE = 32
DEVICE = 'cuda'


# Transform (also check anti_transform in mepis_utils accordingly)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
      [transforms.Resize(320),
      transforms.CenterCrop(300),
      transforms.ToTensor(),
      normalize])

# Load Validation Dataset (No Shuffle)
test_original = DataLoader(TrainingDataset(transform=transform, list='test_set_imagenet_full.json'), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_minival = DataLoader(TrainingDataset(transform=transform, list='imagenet_minival_set_v2.json'), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# Lista de dataloaders
data_loader_list = [test_original, test_minival]
names_loader_list = ['OriginalTestImageNet', 'MinivalImageNet']

def results_per_class(RED_USADA, STARTORBASE):

  # Invocar red neuronal
  wrapper_net = instantiate_network(RED_USADA)
  net = wrapper_net.get_net()
  net.to('cuda')

  # Cargar punto de control red neuronal
  if STARTORBASE == 'BASE':
    NOMBRE_CARPETA_CHECKPOINT = 'I_Extended_Checkpoint'
  elif STARTORBASE == 'START':
    NOMBRE_CARPETA_CHECKPOINT = 'I_StartPoint_Checkpoint'
  elif STARTORBASE == 'METH':
    NOMBRE_CARPETA_CHECKPOINT = 'I_Chosen_Checkpoint'
  elif STARTORBASE == 'PRFIX':
    NOMBRE_CARPETA_CHECKPOINT = 'I_Chosen_II_Checkpoint'
  else:
    raise Exception("Invalid")

  CHECKPOINT = f'{NOMBRE_CARPETA_CHECKPOINT}/{RED_USADA}/checkpoint'
  checkpoint = torch.load(CHECKPOINT)
  net.load_state_dict(checkpoint['model_state_dict'])

  # Directorio almacenamiento
  if STARTORBASE == 'BASE':
    NOMBRE_CARPETA = 'I_Extended_Results_By_Class'
  elif STARTORBASE == 'START':
    NOMBRE_CARPETA = 'I_StartPoint_Results_By_Class'
  elif STARTORBASE == 'METH':
    NOMBRE_CARPETA = 'I_Chosen_Results_By_Class'
  elif STARTORBASE == 'PRFIX':
    NOMBRE_CARPETA = 'I_Chosen_II_Results_By_Class'
  else:
    raise Exception("Invalid")

  DIRECTORIO = f'{NOMBRE_CARPETA}/{RED_USADA}'

  # Neural Network to evaluation mode
  net.eval()

  # Nueva lista de accuracys por cada CHECKPOINT
  lista_de_accuracys = []

  # Testear sobre conjuntos
  for test_loader, name_loader in zip(data_loader_list, names_loader_list):
    running_acc = 0.0
    total_test = len(test_loader.dataset)

    # CSV GEN
    csv_file = open(f'{DIRECTORIO}/{name_loader}_mcnemar_results.csv', 'w')
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
          #print(f'y_gold: {int(elm)}, y_pred: {int(predo)}, eq: {int(elo)}')
        running_acc += torch.sum(idx_pred == Y).item()
        avg_acc = running_acc/total_test*100
      lista_de_accuracys.append(avg_acc)

  return lista_de_accuracys

if __name__ == '__main__':
    parameter_list_names = ["Red Neuronal", "StartXORBase"]
    sys_arguments = [sys.argv[1], sys.argv[2]]
    for p_l_n, sys_arg in zip(parameter_list_names, sys_arguments):
      print(f'Value for {p_l_n} is {sys_arg}')
    results_per_class(sys.argv[1], sys.argv[2])