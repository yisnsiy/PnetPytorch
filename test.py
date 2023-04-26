import matplotlib
from os.path import join
import time
import torch
import math
from torch.utils.data import TensorDataset,DataLoader
from torch import nn
import pandas as pd
import logging
from config import LOG_PATH
import numpy as np
from matplotlib import pyplot as plt
from captum.attr import LayerDeepLift, NeuronDeepLift, DeepLift, IntegratedGradients, LayerIntegratedGradients, NeuronIntegratedGradients

pnet = {"accuracy": 0.7,
        "precision": 0.8,
        "f1": 0.9,
        "recall": 0.5,
        "aupr": 0.6}

col = list(pnet.keys())
data = list([pnet.values()])
results = pd.DataFrame(data=data, index=['pnet'], columns=col)

linear = {"accuracy": 0.6,
        "precision": 0.7,
        "f1": 0.8,
        "recall": 0.4,
        "aupr": 0.5}

col = list(linear.keys())
data = list(linear.values())
results.loc['linear'] = data


cvr = pd.read_csv(join(LOG_PATH, 'cross_valida_result_tra_val_tes.csv'))
print(cvr)
cvr.set_index(['Unnamed: 0'], inplace=True)
print(cvr)
mean = cvr.mean()
std = cvr.std()
print(mean, std)