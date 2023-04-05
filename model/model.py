from utils.animator import Animator
from utils.metrics import Metrics
from utils.general import Accumulator, Timer
from config import models_params, data_params
from model.builder_utils import *
from custom.layer_custom import CustomizedLinear, Diagonal
from data_access.data_access import Data

import torch
from torch import nn, tanh, sigmoid
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from typing import Union, List, Mapping

use_bias = models_params['params']['model_params']['use_bias']
trainable_mask = models_params['params']['model_params']['trainable_mask']


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.get_model_params()
        self.get_model_data()

        print("input dim: ", self.in_size)

        # self.Tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(p=self.model_params['dropout'][0])
        self.dropout2 = nn.Dropout(p=self.model_params['dropout'][1])
        # self.sigmoid = nn.Sigmoid()

        # each node in the next layer is connected to exactly three nodes of the input
        # layer representing mutations, copy number amplification and copy number deletions.
        # sparse layer 27687*9229
        self.h0 = Diagonal(self.in_size, self.maps[0].shape[0], use_bias, trainable_mask)  # sparse layer 27687*9229
        # sparse layer 9227*1387
        self.h1 = CustomizedLinear(self.maps[0].shape[0], self.maps[0].shape[1], use_bias,
                                    np.array(self.maps[0].values), trainable_mask)
        # sparse layer 1387*1066
        self.h2 = CustomizedLinear(self.maps[1].shape[0], self.maps[1].shape[1], use_bias,
                                    np.array(self.maps[1].values), trainable_mask)
        # sparse layer 1066*447
        self.h3 = CustomizedLinear(self.maps[2].shape[0], self.maps[2].shape[1], use_bias,
                                    np.array(self.maps[2].values), trainable_mask)
        # sparse layer 447*147
        self.h4 = CustomizedLinear(self.maps[3].shape[0], self.maps[3].shape[1], use_bias,
                                    np.array(self.maps[3].values), trainable_mask)
        # sparse layer 147*26
        self.h5 = CustomizedLinear(self.maps[4].shape[0], self.maps[4].shape[1], use_bias,
                                    np.array(self.maps[4].values), trainable_mask)
        # self.h0 = nn.Linear(self.in_size, self.maps[0].shape[0])  # sparse layer 27687*9229
        # self.h1 = nn.Linear(self.maps[0].shape[0], self.maps[0].shape[1])  # sparse layer 9227*1387
        # self.h2 = nn.Linear(self.maps[1].shape[0], self.maps[1].shape[1])  # sparse layer 1387*1066
        # self.h3 = nn.Linear(self.maps[2].shape[0], self.maps[2].shape[1])  # sparse layer 1066*447
        # self.h4 = nn.Linear(self.maps[3].shape[0], self.maps[3].shape[1])  # sparse layer 447*147
        # self.h5 = nn.Linear(self.maps[4].shape[0], self.maps[4].shape[1])  # sparse layer 147*26

        self.l1 = nn.Linear(self.maps[0].shape[0], 1)
        self.l2 = nn.Linear(self.maps[0].shape[1], 1)
        self.l3 = nn.Linear(self.maps[1].shape[1], 1)
        self.l4 = nn.Linear(self.maps[2].shape[1], 1)
        self.l5 = nn.Linear(self.maps[3].shape[1], 1)
        self.l6 = nn.Linear(self.maps[4].shape[1], 1)

    def forward(self, input):
        out = tanh(self.h0(input))
        o1 = sigmoid(self.l1(out))
        out = self.dropout1(out)

        out = tanh(self.h1(out))
        o2 = sigmoid(self.l2(out))
        out = self.dropout2(out)

        out = tanh(self.h2(out))
        o3 = sigmoid(self.l3(out))
        out = self.dropout2(out)

        out = tanh(self.h3(out))
        o4 = sigmoid(self.l4(out))
        out = self.dropout2(out)

        out = tanh(self.h4(out))
        o5 = sigmoid(self.l5(out))
        out = self.dropout2(out)

        out = tanh(self.h5(out))
        o6 = sigmoid(self.l6(out))
        out = self.dropout2(out)

        return torch.concat([o1, o2, o3, o4, o5, o6], dim=1)


    def get_model_params(self):
        self.params = models_params['params']
        self.model_params = self.params['model_params']
        self.fitting_params = self.params['fitting_params']
        print("model params: ", self.params)

    def get_model_data(self):
        """get mask matrix that sparseNn have."""
        data = Data(**data_params)
        x, y, info, cols = data.get_data()
        self.in_size = cols.shape[0]
        print('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

        if hasattr(cols, 'levels'):
            genes = cols.levels[0]
        else:
            genes = cols;
        self.feature_names = {}
        self.feature_names['inputs'] = cols
        if self.model_params['n_hidden_layers'] > 0:
            maps = get_layer_maps(genes, self.model_params['n_hidden_layers'], 'root_to_leaf', False)
            self.maps = maps;
        for i, _maps in enumerate(maps):
            self.feature_names[f'h{i}'] = _maps.index

def get_loss(output: torch.Tensor,
             y: torch.Tensor,
             loss: Union[nn.Module, str, List[nn.Module], List[str]],
             loss_weights: Union[torch.Tensor, np.ndarray, List] = None,
             class_weights: Mapping[int, float] = None,
             device=None,
             ) -> torch.Tensor:
    """Return loss value and predicted value in multi output or single output.

    Args:
        output: N*M matrix that N represent batch size and M represent
          number of model's output.
        y: true label in N*1 matrix
        loss: list or single element, if a list, an element in list will word
          in an output of model.
        loss_weights: Optional list or dictionary specifying scalar coefficients
          (Python floats) to weight the loss contributions of different model
          outputs. The loss value that will be minimized by the model will then
          be the weighted sum of all individual losses, weighted by the
          loss_weights coefficients. If a list, it is expected to have a 1:1
          mapping to the model's outputs. If a dict, it is expected to map
          output names (strings) to scalar coefficients.
        class_weights: Optional dictionary mapping class indices (integers) to
          a weight (float) value, used for weighting the loss function (during
          training only). This can be useful to tell the model to "pay more
          attention" to samples from an under-represented class.
        device: GPU or CPU
    Returns:
        return loss value
    """

    # multi output
    if len(output.shape) > 1 and output.shape[1] > 1:
        if type(loss) is not list:
            loss = [loss]
        if len(loss) is not output.shape[1] and len(loss) == 1:
            loss = loss * output.shape[1]
        assert len(loss) == output.shape[1], f'number of gave loss function' \
                                             f'{len(loss)} is not equal number of' \
                                             f'model {output.shpe[1]}.'
        if loss_weights is not None and len(loss_weights) is not output.shape[1]:
            raise RuntimeError(f'loss wight len {len(loss)} is not equal number of'
                               f'model {output.shpe[1]}.')
        if class_weights is not None:
            cw = [[class_weights[true_label.item()]] for true_label in y]
            for i in range(len(loss)):
                loss[i] = nn.BCELoss(weight=torch.tensor(cw)).to(device)
        l = 0.
        for i, loss_fc in enumerate(loss):
            loss_weight = loss_weights[i] if loss_weights is not None else 1.
            l += loss_weight * loss_fc(output[:, i: i + 1], y)
        return l
    # single output
    else:
        return loss(output, y)


def get_probability(output: torch.Tensor) -> torch.Tensor:
    """Return predict value by gave multi or single output.

    Args:
        output: model's output whose shape should is N*M. N represent
          batch size, M represent number of output.

    Returns:
        return predict value.
    """
    # multi output
    if len(output.shape) > 1 and output.shape[1] > 1:
        if output is None:
            raise ValueError('The output must not be empty.')
        if not isinstance(output, torch.Tensor):
            raise TypeError(f'expect tensor, but get {type(output)}')
        return torch.sum(output, dim=1, keepdim=True) / output.shape[1]
        # single output
    else:
        return output


def model_train(net, train_iter: DataLoader, loss=None, optimizer=None,
                test_iter=None, num_epochs: int = 50, scheduler=None,
                loss_weights=None, class_weights=None, device=None,
                ) -> nn.Module:
    """network train model by train_data and evaluate model by eval_data
    each epoch.

    Args:
        net: a model.
        train_iter: training data_access whose type is DataLoader.
        loss: loss function. if this parameter is none, default loss
          function is Mean squared error.
        optimizer: optimizer of model. if this parameter is none, default
          optimizer is Stochastic gradient descent, and learning rate will
          be set 0.05.
        test_iter: it usually uses to evaluate the performance of model
        num_epochs: training round for all training data_access
        scheduler: in order to set dynamically learn rate
        loss_weights: Optional list or dictionary specifying scalar coefficients
          (Python floats) to weight the loss contributions of different model
          outputs. The loss value that will be minimized by the model will then
          be the weighted sum of all individual losses, weighted by the
          loss_weights coefficients. If a list, it is expected to have a 1:1
          mapping to the model's outputs. If a dict, it is expected to map
          output names (strings) to scalar coefficients.
        class_weights: Optional dictionary mapping class indices (integers) to
          a weight (float) value, used for weighting the loss function (during
          training only). This can be useful to tell the model to "pay more
          attention" to samples from an under-represented class.
        device: cpu or gpu

    Returns:
        Net: updated model
    """

    if net is None:
        raise ValueError('The network must not be empty.')
    if train_iter is None:
        raise ValueError('The training data_access must noe be empty.')
    if not isinstance(train_iter, DataLoader):
        raise TypeError(f'expect DataLoader, but get {type(train_iter)}')
    assert (net is not None and train_iter is not None)

    # def init_weights(m):
    #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
    #         nn.init.xavier_uniform_(m.weight)
    #
    # net.apply(init_weights)
    print('training on', device)
    net.to(device)
    if loss is None:
        loss = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    if isinstance(loss, list):
        for loss_fn in loss:
            loss_fn.to(device)

    animator_tr_acc_and_te_acc = Animator(x_label='epoch',
                                          x_lim=[1, num_epochs],
                                          y_lim=[0.3, 1.],
                                          legend=['train acc', 'test acc'])
    animator_tr_loss = Animator(x_label='epoch',
                                x_lim=[1, num_epochs],
                                y_lim=[20, 650],
                                legend=['train loss'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            output = net(X)
            l = get_loss(output, y, loss, loss_weights, class_weights, device)
            y_prob = get_probability(output)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], Metrics.accuracy(y_prob, y), X.shape[0])
            timer.stop()
            if i % 10 == 0 or i == num_batches - 1:
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                animator_tr_loss.add(epoch + (i + 1) / num_batches, train_l)
                animator_tr_acc_and_te_acc.add(epoch + (i + 1) / num_batches,
                                               (train_acc, None))
        scheduler.step()
        test_acc = model_evaluate(net, test_iter, device)
        animator_tr_acc_and_te_acc.add(epoch + 1, (None, test_acc))
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'lr {optimizer.param_groups[0]["lr"]:.6f}, '
              f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')

    print('\n')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}.')
    print(f'training in {metric[2]} samples, '
          f'{metric[2] * num_epochs / timer.sum():.1f} samples/sec')
    print(f'total time spent {timer.sum_human_read()}, on {str(device)}.')

    animator_tr_loss.show()
    animator_tr_acc_and_te_acc.show()
    plt.show()
    return net


def model_evaluate(net, data_iter: DataLoader,
                   device=None) -> float:
    """Compute the accuracy for a model on a dataset.

    Args:
        net: model.
        data_iter: data_access set.
        device: cpu or gpu.

    Returns:
        Float
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            y_prob = get_probability(output)
            metric.add(Metrics.accuracy(y_prob, y), y.numel())
    return metric[0] / metric[1]


def model_predict(net, X_test, device):
    """get predict score only by test set.

    Args:
        net: model.
        X_test: test set.
        device: GPU or CPU.

    Returns:
        model's directed output that usually represent probability.
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    with torch.no_grad():
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.to(device)
        output = net(X_test)
        y_prob = get_probability(output)
    return y_prob
