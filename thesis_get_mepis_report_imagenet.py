import torch
from PIL import Image
from torchvision import transforms
from thesis_entropy_calculator import get_entropy_2 as get_entropy
from thesis_quality_reductions_composed_v_powell_imagenet import *
from scipy import optimize
import time
from dataset_classes_json_generator import ValidationDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt
import csv
import json
import sys

def generate_transform(r, funcion_objetivo):
  nombre_funcion = funcion_objetivo.__name__
  # Caso combined
  if nombre_funcion == 'entropy_of_a_tensor':
    t_p, b_p, l_p, r_p, d_p, q_p = r
    cropping = transforms.Compose([SliceTop(t_p),SliceBottom(b_p),SliceLeft(l_p),SliceRight(r_p),Downsampling(d_p),Quantization(q_p)])
  # Caso crop
  if nombre_funcion == 'entropy_of_a_tensor_crop':
    t_p, b_p, l_p, r_p = r
    cropping = transforms.Compose([SliceTop(t_p),SliceBottom(b_p),SliceLeft(l_p),SliceRight(r_p)])
  # Caso color
  if nombre_funcion == 'entropy_of_a_tensor_quantization':
    q_p = r
    quantization_factor = float(q_p[0]) # Corrección para ndarray
    cropping = transforms.Compose([Quantization(quantization_factor)])
  # Caso resolucion
  if nombre_funcion == 'entropy_of_a_tensor_downsampling':
    d_p = r
    cropping = transforms.Compose([Downsampling(d_p)])
  return cropping


def get_mepis_report(report_folder, report_name, neural_network, test_loader, funcion_objetivo=None, punto_inicial=(0.075, 0.075, 0.075, 0.075, 0.9, 0.6), names_pics={}, max_iteration_number=3):
  
  # Instanciar CSV
  csv_file = open(f'{report_folder}/{report_name}.csv', 'w', encoding='UTF8',newline='')
  writer = csv.writer(csv_file, delimiter=',')
  writer.writerow(['OriginalValue', 'MinimalValue', 'CorrectIncorrect'])

  # Iterar sobre data_loader
  for i, data in enumerate(test_loader):

    # Obtener los datos del dataloader
    X,Y = data
    # Verificar si cls clasifica correctamente
    _, clasificacion_red = torch.max(neural_network(X.to('cuda'))['logits'], dim=1)
    boolean = int(clasificacion_red) == Y
    # Imagen mínima actual (mepi_0, entropia_0)
    X_reduced_to_show = torch.squeeze(X).to('cpu')
    minimal_image_entropy = get_entropy(torch.squeeze(X_reduced_to_show.to('cpu')))
    original_image_entropy = minimal_image_entropy
    # Si la clasificación es correcta se realiza Powell, si no se ignora
    boolean_as_int = 0
    if boolean:
      boolean_as_int = 1

      # Realizar Powell
      start_time = time.time()

      # Función objetivo
      opt_fun = funcion_objetivo(X.to('cuda'), neural_network, int(Y))

      # Punto Inicial
      X_0 = punto_inicial # 0,1,0,1 significa no croppear. Los 1's son para bottom-up y right-left

      # Obtención de parámetros
      r  = optimize.fmin_powell(opt_fun, X_0, maxiter=max_iteration_number, xtol=0.02)

      # Ver imagen clásica
      X_to_show = torch.squeeze(X.to('cpu'))
      plt.imshow(  X_to_show.permute(1, 2, 0)  )

      # Ver imagen reducida
      cropping = generate_transform(r, funcion_objetivo)

      # Mepi y entropia (mepi_1, entropia_1)
      X_reduced_to_show = torch.squeeze(cropping(X.to('cuda'))).to('cpu')
      minimal_image_entropy = get_entropy(torch.squeeze(X_reduced_to_show.to('cpu')))

    # Imprimir valor de entropia nuevo
    # Mostrar imagen
    plt.imshow(  X_reduced_to_show.permute(1, 2, 0)  )
    # Almacenar imagen con el nombre oficial en la carpeta establecida
    # Almacenar entropía
    writer.writerow([original_image_entropy ,minimal_image_entropy, boolean_as_int])
    sys.stdout.write(f'\rProccessed Images: ({i}/{300})')
  csv_file.close()