import torch
from PIL import Image
from torchvision import transforms
from thesis_powell_search_function_imagenet import entropy_of_a_tensor, entropy_of_a_tensor_crop, entropy_of_a_tensor_downsampling, entropy_of_a_tensor_quantization
from scipy import optimize
import time
from dataloaders_for_imagenet import ValidationDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt
import csv
import json
from thesis_get_mepis_report_imagenet import get_mepis_report

# Import system
import sys

# Redes neuronales
from thesis_networks_controller_imagenet import instantiate_network

def run_it(RED_USADA, EXPERIMENTO):
    # Input red
    red = RED_USADA

    # Wrapper red
    wrapper_net = instantiate_network(red)

    # Instanciar red neuronal
    neural_network = wrapper_net.get_net()

    # Transform (also check anti_transform in mepis_utils accordingly)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    # Dataloader
    transform = transforms.Compose(
      [transforms.Resize(320),
      transforms.CenterCrop(300),
      transforms.ToTensor(),
      normalize])
    test_loader = DataLoader(ValidationDataset(transform=transform, list='test_set_imagenet_full.json'),
                            batch_size=1, shuffle=False, num_workers=1)

    # Generar diccionario de nombres  
    f = open('testing_human_original_adjusted.json',)
    names_pics = {}
    data = json.load(f)
    
    for i, el in zip(range(len(data)), data):
        neim = el[0].split("/")[2]
        names_pics[i] = str(neim)
    
    f.close()

    # Mover red a cuda
    neural_network.to('cuda')



    # Directorio Checkpoint
    NOMBRE_CARPETA_CHECKPOINT = 'F_FullImageNetCheckpoints_II'
    CHECKPOINT = f'{NOMBRE_CARPETA_CHECKPOINT}/{RED_USADA}/{EXPERIMENTO}/checkpoint'

    print(f'The checkpoint is: {CHECKPOINT}')
    checkpoint = torch.load(CHECKPOINT)
    neural_network.load_state_dict(checkpoint['model_state_dict'])

    # Directorio de almacenamiento
    NOMBRE_CARPETA = 'F_FullImageNetMEPIS_II'
    DIRECTORIO = f'{NOMBRE_CARPETA}/{RED_USADA}/{EXPERIMENTO}'

    # Modo evaluación (para predecir correctamente y no calcular gradientes)
    neural_network.eval() #<-- importante

    # Listas
    report_names = ["quantization_report"]
    funciones_objetivos = [entropy_of_a_tensor_quantization]
    puntos_iniciales = [(0.1)]

    # Generar archivo reporte
    for report_name, funcion_objetivo, punto_inicial in zip(report_names, funciones_objetivos, puntos_iniciales):
        print(funcion_objetivo.__name__)
        dir_almacenamiento = f'{DIRECTORIO}/{report_name}'
        get_mepis_report(dir_almacenamiento, report_name, neural_network, test_loader, funcion_objetivo, punto_inicial, names_pics, 3)

if __name__ == '__main__':
    parameter_list_names = ["Red Neuronal", "EXPERIMENTO"]
    sys_arguments = [sys.argv[1], sys.argv[2]]
    for p_l_n, sys_arg in zip(parameter_list_names, sys_arguments):
      print(f'Value for {p_l_n} is {sys_arg}')
    run_it(sys.argv[1], sys.argv[2])