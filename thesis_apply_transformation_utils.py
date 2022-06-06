from thesis_quality_reductions_composed import *
from torchvision import transforms

# Nota: SliceRight SliceLeft SliceBottom y SliceTop no necesitan 1-alfa*factor
# Nota: Downsampling y Quantization sí necesitan 1-alfa*factor.
# Nota: Esto se realiza para estandarizar

def apply_transformation_from_list(img,l,alfa=0.1):
    v_1, v_2, v_3, v_4, v_5, v_6 = l
    process = transforms.Compose([Downsampling(1-v_1*alfa), 
                                Quantization(1-v_2*alfa), 
                                SliceLeft(v_3*alfa/2), 
                                SliceRight(v_4*alfa/2), 
                                SliceTop(v_5*alfa/2), 
                                SliceBottom(v_6*alfa/2)])
    return process(img)