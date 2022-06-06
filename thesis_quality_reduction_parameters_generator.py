import sys
from timeit import default_timer as timer

import torch
from torch.utils.data import Dataset
import numpy as np

import torchvision.io

from torchvision import transforms

import itertools

import random

from thesis_persist_as_list import persist_list

from thesis_apply_transformation_utils import apply_transformation_from_list

import json


def list_diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

def produce_reductions_json(net, train_loader, delta_alpha, red_dim = 'Combined', device='cuda'):
  net.to(device)
  total_train = len(train_loader.dataset)
  dictionary = {}

  for e in range(1):  
    tiempo_inicial = timer()
    
    net.eval()

    computo = 5

    for i, data in enumerate(train_loader):

      # Desagregamos los datos y los pasamos a la GPU
      X, Y, Z = data
      X, Y = X.to(device), Y.to(device)

      # x.shape = [B, 3, 32, 32]
      batch_size = X.shape[0]

      # Copiar el X original para no modificarlo
      copy_x = torch.clone(X)
      copy_x.to(device)

      net.eval() # modo evaluación
      mepis_candidatas = torch.clone(copy_x)
      cantidad_de_rondas = int(1/delta_alpha) - 1
      cantidad_de_rondas = 1
      d = {0.50: 1, 0.25: 3, 0.125: 7, 0.0625: 15}
      cantidad_de_rondas = d[delta_alpha]
      #cantidad_de_rondas = 100
      v = [[0,0,0,0,0,0] for i in range(batch_size)]

      # Verficar que aplicar
      if red_dim == 'Combined':
        v_prime = [[True,True,True,True,True,True] for i in range(batch_size)]
        reductions_to_apply = range(6)
      elif red_dim == 'Resolution':
        v_prime = [[True,False,False,False,False,False] for i in range(batch_size)]
        reductions_to_apply = [1]
      elif red_dim == 'Quantization':
        v_prime = [[False,True,False,False,False,False] for i in range(batch_size)]
        reductions_to_apply = [2]
      elif red_dim == 'Crop':
        v_prime = [[False,False,True,True,True,True] for i in range(batch_size)]
        reductions_to_apply = [2, 3, 4, 5]
      else:
        raise Exception(f'Reduccion no valida: {red_dim}')
      # while(...)
      while True:
        with torch.no_grad():
          #print(cantidad_de_rondas)
          mepis_candidatas_antes = torch.clone(mepis_candidatas)
          v3 = v.copy()
          for t in reductions_to_apply:
            v2 = v3.copy()
            indices_reducciones_saltadas = []
            mepis_candidatas_locales = []
            for foto, reductions_made, idx_foto, make_reduction_bool in zip(copy_x, v2, range(batch_size), v_prime):
              if make_reduction_bool[t]:
                reductions_made[t]+=1
                mepis_candidatas_locales.append(apply_transformation_from_list(foto, reductions_made, delta_alpha)) # Función que aplica reducción
              else:
                indices_reducciones_saltadas.append(idx_foto)
                mepis_candidatas_locales.append(foto) # Se añade la foto sin aplicar reducción
            # Obtención de imágenes correctas usando clasificador
            nueva_mepis_candidatas = torch.stack(mepis_candidatas_locales)
            out_dict_3 = net(nueva_mepis_candidatas)
            Y_logits_3 = out_dict_3['logits']
            _, max_idx_3 = torch.max(Y_logits_3, dim=1)
            A = max_idx_3 == Y
            entropy_ok_eval_ok = A
            K = torch.arange(batch_size, dtype=torch.int8)
            K.to(device)
            aux_list = K[entropy_ok_eval_ok].tolist() # Nuevas MEPIs
            if len(aux_list)<1: # Si no hay índices saltarse esta reducción
              continue
            # Eliminar los indices de las saltadas
            aux_list = list_diff(aux_list, indices_reducciones_saltadas)
            #
            tensor_indices_nuevas_mepis = torch.LongTensor(aux_list).to(device)
            # Crear un tensor con las MEPIs obtenidas
            mepis_confirmadas = torch.index_select(nueva_mepis_candidatas, 0, tensor_indices_nuevas_mepis)
            mepis_confirmadas.to(device)
            mepis_candidatas.index_copy_(0, tensor_indices_nuevas_mepis, mepis_confirmadas) # volcar nuevas mepis (mepis_confirmadas) en mepis_candidatas
            # Actualizar v_prime con los Falses si es que no se aplicó la reducción
            aux_list_complement = list_diff(range(batch_size), aux_list)
            for el in aux_list_complement:
              v_prime[el][t] = False
            #
            for el in aux_list:
              v[el] = v2[el] # actualizar lista de reducciones aplicadas
          cantidad_de_rondas-=1
          # Si las imagenes mínimas no cambian en los 6 intentos o se acaban las rondas, its over.
          if torch.equal(mepis_candidatas, mepis_candidatas_antes) or cantidad_de_rondas<1:
            #print(f'BREAK: CANTIDAD_DE_RONDAS: {cantidad_de_rondas}')
            break
      copy_x = mepis_candidatas
      computo+=1
      if computo < 20:
        torch.save(copy_x, f'mepi_test/mepi_de_prueba_carabina_{computo}.pt')
      # Loading
      items = min(total_train, (i+1) * train_loader.batch_size)
      sys.stdout.write(f'\rProccessed Images: ({items}/{total_train})')
      #print(v)
      for reductions_made, file_name in zip(v, Z):
          dictionary[file_name] = reductions_made
          #print(f'Reductions: {reductions_made}, Name: {file_name}')
    # aquí se debería agrega json...
    with open('mepi_test/json_file.json', 'w') as f:
      json.dump(dictionary, f)
    tiempo_final = timer() - tiempo_inicial
    sys.stdout.write(f', Tiempo total: {tiempo_final}\n')
  return tiempo_final