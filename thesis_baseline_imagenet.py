import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from thesis_persist_as_list import *

import torchvision
import torchvision.transforms as transforms
from dataloaders_for_imagenet import *

# Herramientas de entrenamiento
from thesis_training_utils_imagenet import train_for_classification

from checkpoint_utils_v2 import *

# Importar interpolador
from thesis_linear_interpolation import interpolacion

# Importar generador de transformaciones
from thesis_transformation_generator_hw300 import make_transformation_selection

# Redes neuronales
from thesis_networks_controller_imagenet import instantiate_network

# Prueba conjuntos tricky
from thesis_performance_of_sets import test_over_sets


# DEVICE
DEVICE = 'cuda'

def funcion_corrige_name(value):
  if value<10: return f'0{value}'
  return value

# Config seed
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Early Stopping
early_stopping_num = 3


def run_it(dimension, valor_de_alfa, red, lim_epoch):
  EPOCA_LIMITE = int(lim_epoch)
  print(f'Epoca Limite is: {EPOCA_LIMITE}')
  # FILE SYSTEM
  DIMENSION = f'{"Baseline"}' # Combined, Crop, Quantization, Resolution
  DIMENSION = dimension
  
  print(f'Dimension is: {dimension}')
  print(f'Valor de alfa is: {valor_de_alfa}')
  print(f'Red is: {red}')

  EXPERIMENT_MIX = f"Alfa_0{valor_de_alfa.split('.')[1]}"
  if dimension == "Baseline":
    EXPERIMENT_MIX = f'Alfa_100'
  print(f'Exp Mix is: {EXPERIMENT_MIX}')

  NEURAL_NETWORK_NAME = red
  FOLDER_EXPERIMENT = f'F_FullImageNet_Baseline_Epochs/{NEURAL_NETWORK_NAME}/{DIMENSION}/{EXPERIMENT_MIX}'
  FOLDER_REPORTS = f'{FOLDER_EXPERIMENT}/Reports'
  FOLDER_TEST_REPORTS = f'{FOLDER_EXPERIMENT}/TestReports'
  FOLDER_CHECKPOINTS = f'{FOLDER_EXPERIMENT}/Checkpoints'
  FOLDER_MEPIS_REDUCTION = f'{FOLDER_EXPERIMENT}/MEPIS'
  FOLDER_TRAINING_REPORTS = f'{FOLDER_EXPERIMENT}/TrainingReports'

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

  transform = transforms.Compose(
      [transforms.Resize(320),
      transforms.CenterCrop(300),
      transforms.ToTensor(),
      normalize])

  print(f'Training')
  # Definamos algunos hiper-parámetros
  BATCH_SIZE = 32
  EPOCHS = 1
  REPORTS_EVERY = 1
  CICLOS = 100000000

  # Wrapper red
  wrapper_net = instantiate_network(red)

  # Instanciar red neuronal
  net = wrapper_net.get_net()

  # Hiperparametros de la red
  LR = wrapper_net.get_lr()
  optimizer = wrapper_net.get_optimizer()
  criterion = nn.CrossEntropyLoss()
  scheduler = wrapper_net.get_scheduler()

  # Checkpoint
  print(f'Cargando pesos de punto de control')
  print(wrapper_net.get_checkpoint())
  checkpoint = torch.load(wrapper_net.get_checkpoint())
  net.load_state_dict(checkpoint['model_state_dict'])

  print(f'Load optimizer')
  load_optimizer_checkpoint_cuda(optimizer, checkpoint)

  print(f'Load learning rate')
  for g in optimizer.param_groups:
    g['lr'] = wrapper_net.get_checkpoint_lr()

  train_loader = DataLoader(TrainingDataset(transform=transform, list='imagenet_training_set_v2.json'), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=8)
  test_loader = DataLoader(ValidationDataset(transform=transform, list='imagenet_minival_set_v2.json'), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)
  test_set_loader = DataLoader(ValidationDataset(transform=transform, list='testing_tesis_categories_20_class.json'), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

  # Generar reducción
  reducir = False # Inicialización de la variable de reducción
  reduccion = None
  if dimension != 'Baseline':
    alpha = float(valor_de_alfa)
    DIMENSION = dimension
    reducir = True
    reduccion = make_transformation_selection(alpha, selection=DIMENSION)

  # Acc
  acc_val_actual = wrapper_net.get_checkpoint_acc()

  # Lista resultados
  loss_reports = []
  acc_train = []
  acc_valid = []
  epoch_time = []

  # Lista learning rates
  lista_learning_rates = []

  # Inicialización variables early stopping
  early_stopper_watcher = 0

  epoca_actual = 1
  for i in range(CICLOS):

    print(f'Acc val actual es: {acc_val_actual}')

    if epoca_actual >= EPOCA_LIMITE:
      print(f'SE HA LLEGADO A EPOCA LIMITE: {epoca_actual}')
      break

    # Epocas Clasicas
    for idx_classic in range(EPOCHS):
      #print("TRAINING CLASSIC")
      train_loss, acc, e_time = train_for_classification(net, train_loader, 
                                                test_loader, optimizer, 
                                                criterion, lr_scheduler=scheduler, 
                                                epochs=1, reports_every=REPORTS_EVERY, device='cuda',
                                                reducir=reducir, reduccion=reduccion)
      #print("Almacenando listas")
      persist_tuple_prefix((train_loss, acc), f'{FOLDER_REPORTS}/ImageNet-{NEURAL_NETWORK_NAME}-{funcion_corrige_name(epoca_actual)}_CHECKPOINT_CLASSIC_REPORT')

      # Actualizar listas de resultados
      acc_train.append(acc[0][0])
      acc_valid.append(acc[1][0])
      loss_reports.append(train_loss[0])
      epoch_time.append(e_time)

      # Actualizar lista learning rates
      lista_learning_rates.append(scheduler.get_last_lr())

      if acc[1][0] > acc_val_actual:
        acc_val_actual = acc[1][0]
        print(f'Valor de acc_val_actual actualizado a {acc_val_actual}, almacenando punto de control')
        # Almacenar punto de control época clásica
        PATH = f'./{FOLDER_CHECKPOINTS}/ImageNet-{NEURAL_NETWORK_NAME}-{funcion_corrige_name(epoca_actual)}_CHECKPOINT_CLASSIC'
        save_checkpoint_2(net, optimizer, epoca_actual, PATH, scheduler)

      epoca_actual+=1
  
  # Almacenar punto de control época clásica ultimo
  PATH = f'./{FOLDER_CHECKPOINTS}/ImageNet-{NEURAL_NETWORK_NAME}-{funcion_corrige_name(epoca_actual)}_CHECKPOINT_CLASSIC'
  save_checkpoint_2(net, optimizer, epoca_actual, PATH, scheduler)
  
  # Almacenar listas
  persist_list(acc_train, f'{FOLDER_TRAINING_REPORTS}/TrainingAccuracy')
  persist_list(acc_valid, f'{FOLDER_TRAINING_REPORTS}/ValidationAccuracy')
  persist_list(loss_reports, f'{FOLDER_TRAINING_REPORTS}/TrainingLoss')
  persist_list(epoch_time, f'{FOLDER_TRAINING_REPORTS}/EpochsTime')

  # Almacenar lista learning rates
  persist_list(lista_learning_rates, f'{FOLDER_TRAINING_REPORTS}/LearningRates')


if __name__ == '__main__':
    parameter_list_names = ["Dimension", "Valor de Alfa", "Red Neuronal", "Epoca Limite"]
    sys_arguments = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
    for p_l_n, sys_arg in zip(parameter_list_names, sys_arguments):
      print(f'Value for {p_l_n} is {sys_arg}')
    run_it(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])