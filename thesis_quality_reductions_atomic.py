import torch
import torchvision
import PIL

class Quantization(torch.nn.Module):
  """
  Quantization
  """
  def __init__(self, actual, verbose=True, device='cuda'):
    super().__init__()
    self.actual = actual
    self.device = device
    self.colores = int(255 * self.actual)
    if self.colores>255: self.colores = 255
    if self.colores<1: self.colores = 1
    self.espaciado = 1/self.colores
    if verbose: print(f'Cantidad de colores por canal: {self.colores}')
    


  def forward(self, img):
    T = torch.round(img/self.espaciado) * self.espaciado
    T = torch.min(T,torch.ones(T.shape).to(self.device))
    return T

class Downsampling(torch.nn.Module):
  """
  Changes resolution of a given tensor
  """
  def __init__(self, downsampling_percentage, imagenet=True):
    super().__init__()
    if downsampling_percentage<0.01: downsampling_percentage = 0.02
    if downsampling_percentage>1: downsampling_percentage = 1
    self.downsampling_percentage = downsampling_percentage
    if imagenet: self.max_res = 224

  def forward(self, img):
    B, C, H, W = img.shape
    img = torchvision.transforms.Resize((int(H*self.downsampling_percentage), int(W*self.downsampling_percentage)))(img)
    img = torchvision.transforms.Resize(self.max_res)(img)
    return img

class Slice(torch.nn.Module):
  """
  Crops an image from all sides
  """
  def __init__(self, crop_percentage, device='cuda'):
    super().__init__()
    print('gray')
    if crop_percentage<0: crop_percentage = 0
    if crop_percentage>1: crop_percentage = 1
    self.crop_percentage = crop_percentage
    self.device = device

  def forward(self, img):
    _, C, H, W = img.shape
    first = 4 * W
    second = (16 * W * W * self.crop_percentage)**(1/2)
    res = (first + second) / 8
    mask = torch.ones((H,W)).to(self.device)
    crop = int(res)
    crop_2 = int(W-res)
    l = list(range(crop,H))
    r = list(range(crop_2))
    if len(l) == 0: return img
    h = torch.tensor(l).to(self.device)
    h_2 = torch.tensor(r).to(self.device)
    mask.index_fill_(0, h, 0)
    mask.index_fill_(0, h_2, 0)
    mask.index_fill_(1, h_2, 0)
    mask.index_fill_(1, h, 0)
    mask_2 = torch.clone(mask)
    mask_2[mask_2==1] = 2
    mask_2[mask_2==0] = 0.5 # gray
    mask_2[mask_2==2] = 0
    ans = img*mask
    ans = ans+mask_2
    return ans