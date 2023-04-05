import matplotlib
import torch
import math
from torch.utils.data import TensorDataset,DataLoader
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from captum.attr import LayerDeepLift, NeuronDeepLift, DeepLift, IntegratedGradients, LayerIntegratedGradients, NeuronIntegratedGradients

# input_features = 2

# class MyModel(nn.Module):
#
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.fc1 = nn.Linear(input_features, 2, bias=False);
#         self.fc2 = nn.Linear(2, 1, bias=False);
#
#     def forward(self, input):
#         out = self.fc1(input)
#         out = torch.tanh(out)
#         out = self.fc2(out)
#         out = torch.sigmoid(out)
#         return out
#
#     def weight_init(self):
#         self.fc1.weight.data_access = torch.Tensor([[1, 2], [3, 4]])
#         self.fc2.weight.data_access = torch.Tensor([[5, 6]])

# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.fc1 = nn.Linear(2, 1, bias=False);
#
#     def forward(self, X):
#         return torch.sigmoid(self.fc1(X))
#
#     def weight_init(self):
#         self.fc1.weight.data_access = torch.Tensor([[1., 1.]])
#
# net = MyModel()
# net.weight_init()
#
#
# net = MyModel()
# net.weight_init()
#
# X = torch.tensor([[50., 0.]])
# # X = torch.randn(1, input_features)
# print("input is ", X)
# net.eval()
# print("output is ", net(X))
#
# print("use Deeplift and IntegratedGradients-----------------------------------------")
#
# d = DeepLift(net)
# d1 = d.attribute(X)
# print(f"dl fc1:{d1}")
#
# ig = IntegratedGradients(net)
# ig1 = ig.attribute(X)
# print(f"ig fc1:{ig1}")
#
# print("\n\nuse LayerDeeplift and LayerIntegratedGradients-----------------------------------------")
# dl1 = LayerDeepLift(net, net.fc1)
# attribution_fc1 = dl1.attribute(X, target=0, attribute_to_layer_input=True)
# print(f"ldl fc1 in:{attribution_fc1}")
# attribution_fc1 = dl1.attribute(X, target=0, attribute_to_layer_input=False)
# print(f"ldl fc1 out:{attribution_fc1}")
#
# lig = LayerIntegratedGradients(net, net.fc1)
# attribution_lig1 = lig.attribute(X, target=0, attribute_to_layer_input=True)
# print(f"lig fc1 in:{attribution_lig1}")
# attribution_lig2 = dl1.attribute(X, target=0, attribute_to_layer_input=False)
# print(f"lig fc1 out:{attribution_lig2}")

# dl2 = LayerDeepLift(net, net.fc2)
# attribution_fc2 = dl2.attribute(X, target=0, attribute_to_layer_input=True)
# print(f"fc2 in:{attribution_fc2}")
# attribution_fc2 = dl2.attribute(X, target=0, attribute_to_layer_input=False)
# print(f"fc2 out :{attribution_fc2}")

#
# print("\n\nuse NeuronDeeplift and NeuronIntegratedGradients-----------------------------------------")
# ndl1 = NeuronDeepLift(net, net.fc1)
# print(f"fc1 neuron1 in: {ndl1.attribute(inputs=X, neuron_selector=0, attribute_to_neuron_input=True)}")
# print(f"fc1 neuron2 out: {ndl1.attribute(X, 0)}")
#
# # ndl2 = NeuronDeepLift(net, net.fc2)
# # print(f"fc2 neuron1 in: {ndl2.attribute(inputs=X, neuron_selector=0, attribute_to_neuron_input=True)}")
# # print(f"fc2 neuron2 out: {ndl2.attribute(X, 0)}")
#
# nig1 = NeuronIntegratedGradients(net, net.fc1)
# print(f"ig neuron1 in: {ndl1.attribute(inputs=X, neuron_selector=0, attribute_to_neuron_input=True)}")
# print(f"ig neuron2 out: {ndl1.attribute(X, 0)}")

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.a = torch.randn(1,2)
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sum(self.fc1(x))

net1 = net()
print(net1.a)

torch.save(net1, 'testsavemodel.pkl')
net2 = net()
print(net2.a)
net2 = torch.load('testsavemodel.pkl')
print(net2)
print(net2.a)