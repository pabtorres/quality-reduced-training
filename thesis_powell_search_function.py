from torchvision import transforms
from thesis_quality_reductions_composed_v_powell import *
from thesis_entropy_calculator import get_entropy_2 as get_entropy

# La función retorna 1|True si la clasificación es correcta 0|False si no lo es
def fun_predictora(classifier, y_gold, input_tensor, device='cuda'):
  _, max_idx = torch.max(classifier(input_tensor.to(device))['logits'], dim=1)
  if max_idx == y_gold:return True
  else: return False

# Función a optimizar con método de Powell
# Función que reduce en todas las dimensiones
# Guardamos el tensor en la clausura de la función, para siempre conservar el tensor original
def entropy_of_a_tensor(tensor, classifier, y_gold, device='cuda', quant_verbose=False):
  def entropy_cropping(tuple_p):
    fun_value = 999999999
    p = 10000
    top_p, bottom_p, left_p, right_p, res_p, quant_p = tuple_p
    cropping = transforms.Compose([Quantization(quant_p, quant_verbose),Downsampling(res_p),SliceTop(top_p),SliceBottom(bottom_p),SliceLeft(left_p),SliceRight(right_p)])
    cropped_tensor = cropping(tensor)
    entropy = get_entropy(torch.squeeze(cropped_tensor.to('cpu')))
    correct = fun_predictora(classifier, y_gold, cropped_tensor, device)
    if correct:
      fun_value = 0
      fun_value = entropy
    if top_p > 1: fun_value += p*fun_value
    if top_p < 0: fun_value += p*fun_value
    if bottom_p > 1: fun_value += p*fun_value
    if bottom_p < 0: fun_value += p*fun_value
    if left_p > 1: fun_value += p*fun_value
    if left_p < 0: fun_value += p*fun_value
    if right_p > 1: fun_value += p*fun_value
    if right_p < 0: fun_value += p*fun_value
    if res_p > 1: fun_value +=p*fun_value
    if res_p < 0: fun_value += p*fun_value
    if quant_p > 1: fun_value +=p*fun_value
    if quant_p < 0: fun_value += p*fun_value
    return fun_value
  return entropy_cropping

# Función a optimizar con método de Powell
# Función que reduce en la dimensión crop solamente
# Guardamos el tensor en la clausura de la función, para siempre conservar el tensor original
def entropy_of_a_tensor_crop(tensor, classifier, y_gold, device='cuda', quant_verbose=False):
  def entropy_cropping(tuple_p):
    fun_value = 999999999
    p = 10000
    top_p, bottom_p, left_p, right_p = tuple_p
    cropping = transforms.Compose([SliceTop(top_p),SliceBottom(bottom_p),SliceLeft(left_p),SliceRight(right_p)])
    cropped_tensor = cropping(tensor)
    entropy = get_entropy(torch.squeeze(cropped_tensor.to('cpu')))
    correct = fun_predictora(classifier, y_gold, cropped_tensor, device)
    if correct:
      fun_value = 0
      fun_value = entropy
    if top_p > 1: fun_value += p*fun_value
    if top_p < 0: fun_value += p*fun_value
    if bottom_p > 1: fun_value += p*fun_value
    if bottom_p < 0: fun_value += p*fun_value
    if left_p > 1: fun_value += p*fun_value
    if left_p < 0: fun_value += p*fun_value
    if right_p > 1: fun_value += p*fun_value
    if right_p < 0: fun_value += p*fun_value
    return fun_value
  return entropy_cropping

# Función a optimizar con método de Powell
# Función que reduce en la dimensión quantization
# Guardamos el tensor en la clausura de la función, para siempre conservar el tensor original
def entropy_of_a_tensor_quantization(tensor, classifier, y_gold, device='cuda', quant_verbose=False):
  def entropy_cropping(tuple_p):
    fun_value = 999999999
    p = 10000
    quant_p = tuple_p
    quantization_factor = float(quant_p[0]) # Corrección para ndarray
    cropping = transforms.Compose([Quantization(quantization_factor, quant_verbose)])
    cropped_tensor = cropping(tensor)
    entropy = get_entropy(torch.squeeze(cropped_tensor.to('cpu')))
    correct = fun_predictora(classifier, y_gold, cropped_tensor, device)
    if correct:
      fun_value = 0
      fun_value = entropy
    if quant_p > 1: fun_value +=p*fun_value
    if quant_p < 0: fun_value += p*fun_value
    return fun_value
  return entropy_cropping

# Función a optimizar con método de Powell
# Función que reduce en la dimensión downsampling
# Guardamos el tensor en la clausura de la función, para siempre conservar el tensor original
def entropy_of_a_tensor_downsampling(tensor, classifier, y_gold, device='cuda', quant_verbose=False):
  def entropy_cropping(tuple_p):
    fun_value = 999999999
    p = 10000
    res_p= tuple_p
    cropping = transforms.Compose([Downsampling(res_p)])
    cropped_tensor = cropping(tensor)
    entropy = get_entropy(torch.squeeze(cropped_tensor.to('cpu')))
    correct = fun_predictora(classifier, y_gold, cropped_tensor, device)
    if correct:
      fun_value = 0
      fun_value = entropy
    if res_p > 1: fun_value +=p*fun_value
    if res_p < 0: fun_value += p*fun_value
    return fun_value
  return entropy_cropping