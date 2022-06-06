from thesis_quality_reductions_hw300 import *
from torchvision import transforms

def make_transformation(alpha):
    return eval(f'transforms.Compose([Downsampling({alpha}), Quantization({alpha}), Slice({alpha}),])')

def make_transformation_slice(alpha):
    return eval(f'transforms.Compose([Slice({alpha}),])')

def make_transformation_downsampling(alpha):
    return eval(f'transforms.Compose([Downsampling({alpha}),])')

def make_transformation_quantization(alpha):
    return eval(f'transforms.Compose([Quantization({alpha}),])')

def make_transformation_selection(alpha, selection):
    if selection=='Combined':
        return make_transformation(alpha)
    elif selection=='Crop':
        return make_transformation_slice(alpha)
    elif selection=='Resolution':
        return make_transformation_downsampling(alpha)
    elif selection=='Quantization':
        return make_transformation_quantization(alpha)
    else:
        raise Exception("Reducción no válida")