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
from dataset_classes_json_generator import *

# Herramientas de entrenamiento
from thesis_training_utils import train_for_classification

from thesis_checkpoint_utils import *

# Importar interpolador
from thesis_linear_interpolation import interpolacion

# Importar generador de transformaciones
from thesis_transfomation_generator import make_transformation_selection

# Redes neuronales
from thesis_networks_controller import instantiate_network

from prueba_sobre_conjuntos_v2 import test_over_sets

# Generador csv resultados por clase
from thesis_results_per_class import results_per_class


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


def run_it(dimension, valor_de_alfa, red, input_folder):
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

  input_folder = f'F_Baselines_{input_folder}'

  NEURAL_NETWORK_NAME = red
  FOLDER_EXPERIMENT = f'Results_Baselines/{input_folder}/{NEURAL_NETWORK_NAME}/{DIMENSION}/{EXPERIMENT_MIX}'
  FOLDER_REPORTS = f'{FOLDER_EXPERIMENT}/Reports'
  FOLDER_TEST_REPORTS = f'{FOLDER_EXPERIMENT}/TestReports'
  FOLDER_CHECKPOINTS = f'{FOLDER_EXPERIMENT}/Checkpoints'
  FOLDER_MEPIS_REDUCTION = f'{FOLDER_EXPERIMENT}/MEPIS'
  FOLDER_TRAINING_REPORTS = f'{FOLDER_EXPERIMENT}/TrainingReports'


  transform = transforms.Compose(
      [transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),])

  print(f'Training')
  # Definamos algunos hiper-parámetros
  BATCH_SIZE = 32
  EPOCHS = 1
  REPORTS_EVERY = 1
  CICLOS = 80

  # Wrapper red
  wrapper_net = instantiate_network(red)

  # Instanciar red neuronal
  net = wrapper_net.get_net()

  # Hiperparametros de la red
  LR = wrapper_net.get_lr()
  optimizer = wrapper_net.get_optimizer()
  criterion = nn.CrossEntropyLoss()
  scheduler = wrapper_net.get_scheduler()

  train_loader = DataLoader(TrainingDataset(transform=transform, list='humanet_training_set_v3_f.json'), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=8)
  test_loader = DataLoader(ValidationDataset(transform=transform, list='humanet_minival_set_v3_f.json'), batch_size=BATCH_SIZE,
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
  acc_val_actual = 0.0

  # Lista resultados
  loss_reports = []
  acc_train = []
  acc_valid = []
  epoch_time = []

  # Lista resultados
  acc_valid_2 = []
  acc_color = []
  acc_combined = []
  acc_crop = []
  acc_resolution = []

  # Lista learning rates
  lista_learning_rates = []

  # Inicialización variables early stopping
  early_stopper_watcher = 0

  epoca_actual = 1
  for i in range(CICLOS):

    if early_stopper_watcher == early_stopping_num:
      print(f'Early Stopped in epoch: {epoca_actual}')
      break

    # Epocas Clasicas
    for idx_classic in range(EPOCHS):
      train_loss, acc, e_time = train_for_classification(net, train_loader, 
                                                test_loader, optimizer, 
                                                criterion, lr_scheduler=scheduler, 
                                                epochs=1, reports_every=REPORTS_EVERY, device='cuda',
                                                reducir=reducir, reduccion=reduccion)
      persist_tuple_prefix((train_loss, acc), f'{FOLDER_REPORTS}/ImageNet-{NEURAL_NETWORK_NAME}-{funcion_corrige_name(epoca_actual)}_CHECKPOINT_CLASSIC_REPORT')

      # Actualizar listas de resultados
      acc_train.append(acc[0][0])
      acc_valid.append(acc[1][0])
      loss_reports.append(train_loss[0])
      epoch_time.append(e_time)

      # Actualizar las otras listas también
      lista_de_accuracys = test_over_sets(net)
      acc_valid_2.append(lista_de_accuracys[0])
      acc_color.append(lista_de_accuracys[1])
      acc_combined.append(lista_de_accuracys[2])
      acc_crop.append(lista_de_accuracys[3])
      acc_resolution.append(lista_de_accuracys[4])

      # Actualizar lista learning rates
      lista_learning_rates.append(scheduler.get_last_lr())

      # Actualizar watcher
      early_stopper_watcher += 1

      if acc[1][0] > acc_val_actual:
        early_stopper_watcher = 0
        acc_val_actual = acc[1][0]
        print(f'Valor de acc_val_actual actualizado a {acc_val_actual}, almacenando punto de control')
        # Almacenar punto de control época clásica
        PATH = f'./{FOLDER_CHECKPOINTS}/ImageNet-{NEURAL_NETWORK_NAME}-{funcion_corrige_name(epoca_actual)}_CHECKPOINT_CLASSIC'
        save_checkpoint_2(net, optimizer, epoca_actual, PATH, scheduler)

      if early_stopper_watcher == early_stopping_num:
        print(f'Early Stopped in epoch: {epoca_actual}')
        break


      epoca_actual+=1
  
  # Almacenar listas
  persist_list(acc_train, f'{FOLDER_TRAINING_REPORTS}/TrainingAccuracy')
  persist_list(acc_valid, f'{FOLDER_TRAINING_REPORTS}/ValidationAccuracy')
  persist_list(loss_reports, f'{FOLDER_TRAINING_REPORTS}/TrainingLoss')
  persist_list(epoch_time, f'{FOLDER_TRAINING_REPORTS}/EpochsTime')

  # Almacenar las otras listas también
  persist_list(acc_color, f'{FOLDER_TRAINING_REPORTS}/TestColor')
  persist_list(acc_combined, f'{FOLDER_TRAINING_REPORTS}/TestCombined')
  persist_list(acc_crop, f'{FOLDER_TRAINING_REPORTS}/TestCrop')
  persist_list(acc_resolution, f'{FOLDER_TRAINING_REPORTS}/TestResolution')

  # Almacenar lista learning rates
  persist_list(lista_learning_rates, f'{FOLDER_TRAINING_REPORTS}/LearningRates')

  # Almacenar csv con resultados finales
  results_per_class(net, FOLDER_TRAINING_REPORTS)


if __name__ == '__main__':
    parameter_list_names = ["Dimension", "Valor de Alfa", "Red Neuronal", "Carpeta de Almacenamiento"]
    sys_arguments = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
    for p_l_n, sys_arg in zip(parameter_list_names, sys_arguments):
      print(f'Value for {p_l_n} is {sys_arg}')
    run_it(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])