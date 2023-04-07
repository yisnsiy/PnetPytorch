import matplotlib
import torch
import math
from torch.utils.data import TensorDataset,DataLoader
from torch import nn
import pandas as pd
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
print(data, col)
results = pd.DataFrame(data=data, index=['pnet'], columns=col)

linear = {"accuracy": 0.6,
        "precision": 0.7,
        "f1": 0.8,
        "recall": 0.4,
        "aupr": 0.5}

col = list(linear.keys())
data = list(linear.values())
results.loc['linear'] = data

print(results)