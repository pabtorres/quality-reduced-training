import torchvision.io, sys
from torchvision import transforms
import torch
import os

# Transformación inversa
inv_trans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                lambda x: x*255
                               ])

# Calcular peso en disco en formato PNG
def get_entropy(input_tensor):
  k = inv_trans(input_tensor)
  a = torchvision.io.encode_png(k.type(torch.uint8)) 
  len(a)
  return sys.getsizeof(a.storage())

def get_entropy_2(input_tensor):
  torchvision.utils.save_image(input_tensor, 'aux_image_get_entropy.png')
  return os.stat('aux_image_get_entropy.png').st_size