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

from thesis_checkpoint_utils import *

# Importar interpolador
from thesis_linear_interpolation import interpolacion

# Importar generador de transformaciones
from thesis_transformation_generator_hw300 import make_transformation_selection

# Redes neuronales
from thesis_networks_controller_imagenet import instantiate_network

# Prueba conjuntos
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


def run_it(dimension, delta_alfa, red):
  # FILE SYSTEM
  DIMENSION = dimension
  
  print(f'Dimension is: {dimension}')
  print(f'Valor de alfa is: {delta_alfa}')
  print(f'Red is: {red}')

  D_ALFA = f'DeltaAlfa_0{delta_alfa[2:]}'

  NEURAL_NETWORK_NAME = red
  FOLDER_EXPERIMENT = f'FullImageNetRollingStartFixed/{NEURAL_NETWORK_NAME}/{DIMENSION}/{D_ALFA}'
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
  EPOCHS = 80
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

  # Dataloaders
  train_loader = DataLoader(TrainingDataset(transform=transform, list='imagenet_training_set_v2.json'), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=8)
  test_loader = DataLoader(ValidationDataset(transform=transform, list='imagenet_minival_set_v2.json'), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)
  test_set_loader = DataLoader(ValidationDataset(transform=transform, list='testing_tesis_categories_20_class.json'), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

  # Inicialización variables reducción
  d_alpha = float(delta_alfa)
  DIMENSION = dimension

  # Acc
  # Cada red entrega ese valor
  acc_val_actual = wrapper_net.get_checkpoint_acc()
  acc_val_anterior = wrapper_net.get_checkpoint_acc()

  # Lista resultados
  loss_reports = []
  acc_train = []
  acc_valid = []
  epoch_time = []
  acc_valid_2 = []
  acc_color = []
  acc_combined = []
  acc_crop = []
  acc_resolution = []

  # Inicio época clásica, reducida, round y creacion checkpoint
  checkpoint_creation = []
  begin_round = []
  alpha_of_round = []
  checkpoint_load = []

  # Lista de resultados sin valores omitidos
  loss_reports_o = []
  acc_train_o = []
  acc_valid_o = []
  epoch_time_o = []
  acc_valid_2_o = []
  acc_color_o = []
  acc_combined_o = []
  acc_crop_o = []
  acc_resolution_o = []
  alfa_values_o = []

  # Almacenar tipo de época
  tipo_clasica = []
  tipo_reducida = []

  # Inicialización variables early stopping
  early_stopper_watcher = 0

  # Inicialización alfa actual
  alpha_actual = 1

  # Inicialización checkpoint
  mejor_modelo = wrapper_net.get_checkpoint()

  # Inicialización lr checkpoint
  mejor_modelo_lr = wrapper_net.get_checkpoint_lr()

  epoca_actual = 1
  for i in range(CICLOS):

    # Almacenar inicio round
    begin_round.append(epoca_actual)

    # Reinicialización variables early stopping
    early_stopper_reduced = 0
    early_stopper_classic = 0

    alpha_actual -= d_alpha
    
    # Antes de crear una transformación validar valor de alfa actual
    if alpha_actual <= 0.0:
      alpha_actual = 1.0 - d_alpha

    # Almacenar valor de alfa actual
    alpha_of_round.append(alpha_actual)
    
    reducir = True
    reduccion = make_transformation_selection(alpha_actual, selection=DIMENSION)


    if early_stopper_watcher >= early_stopping_num:
      # No se ejecutó la nueva época por lo que la época actual se resta 1-
      print(f'Early Stopped in epoch: {epoca_actual-1}')
      break

    flag_augment_stopper = True
    acc_val_local = 0.0
    cantidad_de_epocas_consideradas = 0

    for idx_classic in range(EPOCHS):
      # Epocas Reducidas
      print(f'r: {alpha_actual}')
      tipo_reducida.append(1)
      tipo_clasica.append(0)
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

      early_stopper_reduced += 1

      if acc[1][0] > acc_val_local:
        early_stopper_reduced = 0
        acc_val_local = acc[1][0]

      if acc[1][0] > acc_val_actual:
        flag_augment_stopper = False
        early_stopper_watcher = 0
        early_stopper_reduced = 0
        acc_val_actual = acc[1][0]
        print(f'Valor de acc_val_actual actualizado a {acc_val_actual}, almacenando punto de control')
        # Almacenar punto de control época clásica
        PATH = f'./{FOLDER_CHECKPOINTS}/ImageNet-{NEURAL_NETWORK_NAME}-{funcion_corrige_name(epoca_actual)}_CHECKPOINT_CLASSIC'
        save_checkpoint(net, optimizer, epoca_actual, PATH)
        mejor_modelo = PATH
        mejor_modelo_lr = scheduler.get_last_lr()[0]
        checkpoint_creation.append(epoca_actual)

      if early_stopper_reduced == early_stopping_num:
        print(f'Early Stopped in epoch: {epoca_actual}')
        epoca_actual+=1 # Aumentar valor para la época clásica siguiente
        cantidad_de_epocas_consideradas += 1
        break


      epoca_actual+=1
      cantidad_de_epocas_consideradas += 1

    for idx_classic in range(EPOCHS):
      # Épocas Clásicas
      print(f'c: {alpha_actual}')
      tipo_reducida.append(0)
      tipo_clasica.append(1)
      train_loss, acc, e_time = train_for_classification(net, train_loader, 
                                                test_loader, optimizer, 
                                                criterion, lr_scheduler=scheduler, 
                                                epochs=1, reports_every=REPORTS_EVERY, device='cuda',
                                                reducir=False, reduccion=None)
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

      early_stopper_classic += 1

      if acc[1][0] > acc_val_local:
        early_stopper_classic = 0
        acc_val_local = acc[1][0]

      if acc[1][0] > acc_val_actual:
        flag_augment_stopper = False
        early_stopper_watcher = 0
        early_stopper_classic = 0
        acc_val_actual = acc[1][0]
        print(f'Valor de acc_val_actual actualizado a {acc_val_actual}, almacenando punto de control')
        # Almacenar punto de control época clásica
        PATH = f'./{FOLDER_CHECKPOINTS}/ImageNet-{NEURAL_NETWORK_NAME}-{funcion_corrige_name(epoca_actual)}_CHECKPOINT_CLASSIC'
        save_checkpoint(net, optimizer, epoca_actual, PATH)
        mejor_modelo = PATH
        mejor_modelo_lr = scheduler.get_last_lr()[0]
        checkpoint_creation.append(epoca_actual)
      
      if early_stopper_classic == early_stopping_num:
        print(f'Early Stopped in epoch: {epoca_actual}')
        epoca_actual+=1 # Aumentar valor para la época reducida siguiente
        cantidad_de_epocas_consideradas += 1
        break


      epoca_actual+=1
      cantidad_de_epocas_consideradas += 1

    # Actualizar watcher
    # Caso: No aumenta rendimiento en el ciclo
    if flag_augment_stopper:
      early_stopper_watcher += 1

    # Caso: Aumenta el rendimiento en el ciclo, guardar las épocas del ciclo
    if not flag_augment_stopper:
      # Actualizar listas de resultados
      acc_train_o += acc_train[-cantidad_de_epocas_consideradas:]
      acc_valid_o += acc_valid[-cantidad_de_epocas_consideradas:]
      loss_reports_o += loss_reports[-cantidad_de_epocas_consideradas:]
      epoch_time_o += epoch_time[-cantidad_de_epocas_consideradas:]

      # Actualizar las otras listas también
      acc_valid_2_o += acc_valid_2[-cantidad_de_epocas_consideradas:]
      acc_color_o += acc_color[-cantidad_de_epocas_consideradas:]
      acc_combined_o += acc_combined[-cantidad_de_epocas_consideradas:]
      acc_crop_o += acc_crop[-cantidad_de_epocas_consideradas:]
      acc_resolution_o += acc_resolution[-cantidad_de_epocas_consideradas:]

      # Actualizar lista alfas
      alfa_values_o += [alpha_actual]

    # Si no hubo cambios en el accuracy aumenta el contador
    if acc_val_anterior == acc_val_actual:
      print(f'Valor de la flag: {flag_augment_stopper} debiese ser False')
      # Buscar checkpoint de mejor modelo.
      print(f'Best checkpoint is in: {mejor_modelo}')
      checkpoint = torch.load(mejor_modelo)
      net.load_state_dict(checkpoint['model_state_dict'])
      load_optimizer_checkpoint_cuda(optimizer, checkpoint)
      for g in optimizer.param_groups:
        g['lr'] = mejor_modelo_lr
      #early_stopper_watcher += 1 # Se comenta para no aumentar dos veces
      checkpoint_load.append(epoca_actual)

    # Si se actualizó el valor entonces se reinicia el contador y actualizar acc_val_anterior
    else:
      early_stopper_watcher = 0
      acc_val_anterior = acc_val_actual

  
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

  # Almacenar valores saltando valores no usados
  # Alfa values
  persist_list(alfa_values_o, f'{FOLDER_TRAINING_REPORTS}/OAlfaValues')

  # Almacenar listas
  persist_list(acc_train_o, f'{FOLDER_TRAINING_REPORTS}/OTrainingAccuracy')
  persist_list(acc_valid_o, f'{FOLDER_TRAINING_REPORTS}/OValidationAccuracy')
  persist_list(loss_reports_o, f'{FOLDER_TRAINING_REPORTS}/OTrainingLoss')
  persist_list(epoch_time_o, f'{FOLDER_TRAINING_REPORTS}/OEpochsTime')

  # Almacenar las otras listas también
  persist_list(acc_color_o, f'{FOLDER_TRAINING_REPORTS}/OTestColor')
  persist_list(acc_combined_o, f'{FOLDER_TRAINING_REPORTS}/OTestCombined')
  persist_list(acc_crop_o, f'{FOLDER_TRAINING_REPORTS}/OTestCrop')
  persist_list(acc_resolution_o, f'{FOLDER_TRAINING_REPORTS}/OTestResolution')

  # Almacenar listas
  persist_list(checkpoint_creation, f'{FOLDER_TRAINING_REPORTS}/CheckpointCreation')
  persist_list(begin_round, f'{FOLDER_TRAINING_REPORTS}/BeginRoundEpoch')
  persist_list(alpha_of_round, f'{FOLDER_TRAINING_REPORTS}/AlphaOfTheRound')
  persist_list(checkpoint_load, f'{FOLDER_TRAINING_REPORTS}/CheckpointLoads')

  # Almacenar lista tipo de epoca
  persist_list(tipo_clasica, f'{FOLDER_TRAINING_REPORTS}/ListaTipoDeEpocaClasica')
  persist_list(tipo_reducida, f'{FOLDER_TRAINING_REPORTS}/ListaTipoDeEpocaReducida')



if __name__ == '__main__':
    parameter_list_names = ["Dimension", "Valor de Alfa", "Red Neuronal"]
    sys_arguments = [sys.argv[1], sys.argv[2], sys.argv[3]]
    for p_l_n, sys_arg in zip(parameter_list_names, sys_arguments):
      print(f'Value for {p_l_n} is {sys_arg}')
    run_it(sys.argv[1], sys.argv[2], sys.argv[3])