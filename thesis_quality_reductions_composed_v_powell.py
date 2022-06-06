import torch
import torchvision
import PIL

class Quantization(torch.nn.Module):
  """
  Quantization
  """
  def __init__(self, actual, verbose=False, device='cuda'):
    super().__init__()
    self.device = device
    self.actual = actual
    actual=self.actual*255.0
    boxes=[]
    bits=int(round(actual))
    if bits>255:
      bits=1
    if bits<1:
      bits=1
    stair=int(round(255/bits))
    for i in range(0,256,stair):
      boxes.append(float(i)/255)
    self.espaciado = 1/len(boxes)
    if verbose: print(f'Cantidad de colores por canal: {len(boxes)}')
    


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
    _, C, H, W = img.shape
    img = torchvision.transforms.Resize((int(H*self.downsampling_percentage), int(W*self.downsampling_percentage)))(img)
    img = torchvision.transforms.Resize(self.max_res)(img)
    return img

class Slice(torch.nn.Module):
  """
  Crops an image from all sides
  """
  def __init__(self, crop_percentage, device='cuda'):
    super().__init__()
    if crop_percentage<0: crop_percentage = 0
    if crop_percentage>1: crop_percentage = 1
    self.crop_percentage = crop_percentage
    self.device = device

  def forward(self, img):
    C, H, W = img.shape
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
    mask_2[mask_2==0] = 0.5
    mask_2[mask_2==2] = 0
    ans = img*mask
    ans = ans+mask_2
    return ans

class SliceTop(torch.nn.Module):
  """
  Cropps an image from top to bottom
  """
  def __init__(self, crop_percentage, device='cuda'):
    super().__init__()
    if crop_percentage<0: crop_percentage = 0
    if crop_percentage>1: crop_percentage = 1
    self.crop_percentage = crop_percentage
    self.device = device

  def forward(self, img):
    _, C, H, W = img.shape
    mask = torch.ones((H,W)).to(self.device)
    crop = int(H*self.crop_percentage)
    l = list(range(crop))
    if len(l) == 0: return img
    h = torch.tensor(l).to(self.device)
    mask.index_fill_(0, h, 0)
    mask_2 = torch.clone(mask)
    mask_2[mask_2==1] = 2
    mask_2[mask_2==0] = 0.5
    mask_2[mask_2==2] = 0
    ans = img*mask
    ans = ans+mask_2
    return ans

class SliceBottom(torch.nn.Module):
  """
  Cropps an image from bottom to top
  """
  def __init__(self, crop_percentage, device='cuda'):
    super().__init__()
    if crop_percentage<0: crop_percentage = 0
    if crop_percentage>1: crop_percentage = 1
    self.crop_percentage = crop_percentage
    self.device = device

  def forward(self, img):
    _, C, H, W = img.shape
    mask = torch.ones((H,W)).to(self.device)
    crop = int(H-H*self.crop_percentage)
    l = list(range(crop,H))
    if len(l) == 0: return img
    h = torch.tensor(l).to(self.device)
    mask.index_fill_(0, h, 0)
    mask_2 = torch.clone(mask)
    mask_2[mask_2==1] = 2
    mask_2[mask_2==0] = 0.5
    mask_2[mask_2==2] = 0
    ans = img*mask
    ans = ans+mask_2
    return ans

class SliceLeft(torch.nn.Module):
  """
  Cropps an image from left to right
  """
  def __init__(self, crop_percentage, device='cuda'):
    super().__init__()
    if crop_percentage<0: crop_percentage = 0
    if crop_percentage>1: crop_percentage = 1
    self.crop_percentage = crop_percentage
    self.device = device

  def forward(self, img):
    _, C, H, W = img.shape
    mask = torch.ones((H,W)).to(self.device)
    crop = int(W*self.crop_percentage)
    l = list(range(crop))
    if len(l) == 0: return img
    v = torch.tensor(l).to(self.device)
    mask.index_fill_(1, v, 0)
    mask_2 = torch.clone(mask)
    mask_2[mask_2==1] = 2
    mask_2[mask_2==0] = 0.5
    mask_2[mask_2==2] = 0
    ans = img*mask
    ans = ans+mask_2
    return ans

class SliceRight(torch.nn.Module):
  """
  Cropps an image from right to left
  """
  def __init__(self, crop_percentage, device='cuda'):
    super().__init__()
    if crop_percentage<0: crop_percentage = 0
    if crop_percentage>1: crop_percentage = 1
    self.crop_percentage = crop_percentage
    self.device = 'cuda'

  def forward(self, img):
    _, C, H, W = img.shape
    mask = torch.ones((H,W)).to(self.device)
    crop = int(W-W*self.crop_percentage)
    l = list(range(crop,W))
    if len(l) == 0: return img
    v = torch.tensor(l).to(self.device)
    mask.index_fill_(1, v, 0)
    mask_2 = torch.clone(mask)
    mask_2[mask_2==1] = 2
    mask_2[mask_2==0] = 0.5
    mask_2[mask_2==2] = 0
    ans = img*mask
    ans = ans+mask_2
    return ans