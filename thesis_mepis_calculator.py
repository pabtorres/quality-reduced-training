import torch
from PIL import Image
from torchvision import transforms
from thesis_powell_search_function import entropy_of_a_tensor, entropy_of_a_tensor_crop, entropy_of_a_tensor_downsampling, entropy_of_a_tensor_quantization
from scipy import optimize
import time
from dataset_classes_json_generator import ValidationDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt
import csv
import json
from thesis_get_mepis_report import get_mepis_report

# Import system
import sys

# Redes neuronales
from thesis_networks_controller import instantiate_network

def run_it(RED_USADA, METODOLOGIA, ADAPTIVEORFIXED, DIMENSION):
    # Input red
    red = RED_USADA

    # Wrapper red
    wrapper_net = instantiate_network(red)

    # Instanciar red neuronal
    neural_network = wrapper_net.get_net()

    # Dataloader
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    test_loader = DataLoader(ValidationDataset(transform=transform, list='humanet_test_set_one_fifty.json'),
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


    # Directorio de almacenamiento
    NOMBRE_CARPETA = 'F_EntropyMeasures_II'
    print(NOMBRE_CARPETA)

    # Directorio Checkpoint
    NOMBRE_CARPETA_CHECKPOINT = 'F_EMCheckpoints'


    DIRECTORIO = f'{NOMBRE_CARPETA}/{RED_USADA}/{METODOLOGIA}/{ADAPTIVEORFIXED}/{DIMENSION}'

    # Cargar punto de control
    CHECKPOINT = f'{NOMBRE_CARPETA_CHECKPOINT}/{RED_USADA}/{METODOLOGIA}/{ADAPTIVEORFIXED}/{DIMENSION}/checkpoint'
    print(f'The checkpoint is: {CHECKPOINT}')
    checkpoint = torch.load(CHECKPOINT)
    neural_network.load_state_dict(checkpoint['model_state_dict'])

    # Modo evaluación (para predecir correctamente y no calcular gradientes)
    neural_network.eval() #<-- importante

    # Listas
    report_names = ["combined_report", "crop_report", "downsampling_report", "quantization_report"]
    funciones_objetivos = [entropy_of_a_tensor, entropy_of_a_tensor_crop, entropy_of_a_tensor_downsampling, entropy_of_a_tensor_quantization]
    puntos_iniciales = [(0.075, 0.075, 0.075, 0.075, 0.95, 0.1), (0.075, 0.075, 0.075, 0.075), (0.95), (0.1)]

    # Generar archivo reporte
    for report_name, funcion_objetivo, punto_inicial in zip(report_names, funciones_objetivos, puntos_iniciales):
        print(funcion_objetivo.__name__)
        dir_almacenamiento = f'{DIRECTORIO}/{report_name}'
        get_mepis_report(dir_almacenamiento, report_name, neural_network, test_loader, funcion_objetivo, punto_inicial, names_pics, 3)

if __name__ == '__main__':
    parameter_list_names = ["Red Neuronal", "Metodología", "AdaptiveXORFixed", "Dimension"]
    sys_arguments = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
    for p_l_n, sys_arg in zip(parameter_list_names, sys_arguments):
      print(f'Value for {p_l_n} is {sys_arg}')
    run_it(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])