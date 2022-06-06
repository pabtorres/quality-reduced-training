# Training Convolutional Neural Networks for Image Classification with Quality-Reduced Examples
Repository for my thesis scripts for Training Convolutional Neural Networks with Quality Reduced Examples

Prerequisites
- ImageNet
- Python 3

Hardware we used:

The experiments were performed on a computer with an 8 core AMD Ryzen 7 1800X CPU, a 64 bit architecture, 94 GBs of RAM, and an Nvidia RTX3090 GPU with 24 GB of G6X memory. The operating system is Linux with the Arch distribution. The Neural Network models are provided by the Pytorch Python Library, version 1.9.0+cu111 and Nvidia CUDA version 11.5.

File system:

```
root 
│
└───ImageNet
│   │
│   └───n04263257
│       │   n04263257_2287.JPEG
│       │   n04263257_9063.JPEG
│       │   ...
│   
└───validationImageNet
    │   ILSVRC2012_val_00006350.JPEG
    │   ILSVRC2012_val_00010745.JPEG
    |   ...
```

Install requirements

`pip install -r thesis_requirements.txt`


How to use?

Depending on the script to use, you can decide the dimension of quality-redution, the reduction step number, neural network, and folder depending on your file-system, for example:

Command for Start Point Runner:

`python thesis_start_point_runner.py Baseline 1.00 effnetb3 folder`

Command for Paired Epochs Adaptive:

`python thesis_paired_epochs_adaptive.py Combined 0.125 effnetb3 folder`

Acknowledgment

The base training methodology (baseline) is based on [CC6204](https://github.com/dccuchile/CC6204/blob/master/2020/tareas/tarea4/utils.py) course's convolutional networks training utilities. Our methodologies extend the base training methodology to implement the quality reduced training.