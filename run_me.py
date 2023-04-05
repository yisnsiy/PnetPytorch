from os.path import join
from data_access.Pnetdata import Pnetdata
from model.model import Model, model_train, model_predict
from config import RESULT_PATH, debug, save_res, loss_weights, models_params
from utils.general import try_gpu, create_data_iterator
from utils.metrics import Metrics
from custom import interpret_model

import torch
from torch import optim, nn
from torch.optim import lr_scheduler
from torchkeras import summary

class_weights = {0: 0.7458410351201479, 1: 1.5169172932330828}
batch_size = 50
epochs = 300
penalty = 0.001

# """
device = try_gpu(3)
device = torch.device('cpu')
if debug is True:
    epochs = 2
    batch_size = 5


# get data_access
print("------loading data_access------")
all_data = Pnetdata()


# build neural network
print("------build model------")
model = Model()
if debug is False:
    summary(model, input_shape=(27687, ))


# prepare data set
print("------prepare data set------")
train_iter = create_data_iterator(X=all_data.x_train, y=all_data.y_train,
                                     batch_size=batch_size, shuffle=True,
                                     data_type=torch.float32, )
valid_iter = create_data_iterator(X=all_data.x_validate_, y=all_data.y_validate_,
                                     batch_size=batch_size, shuffle=True,
                                     data_type=torch.float32)


# train model
print("------train model------")
loss_fn = [nn.BCELoss()] * 6
optimizer = optim.Adam(model.parameters(), lr=models_params['params']['fitting_params']['lr'],
                       weight_decay=penalty)
scheduler = lr_scheduler.StepLR(optimizer,
                                step_size=models_params['params']['fitting_params']['reduce_lr_after_nepochs']['epochs_drop'],
                                gamma=models_params['params']['fitting_params']['reduce_lr_after_nepochs']['drop'])
net = model_train(model, train_iter, loss_fn, optimizer,  valid_iter, epochs,
                  scheduler, loss_weights, class_weights, device)


# evaluate performance
print("------Evaluate performance------")
X_test = torch.tensor(all_data.x_test_, dtype=torch.float32)
y_prob = model_predict(net, X_test, device)
saving_dir = RESULT_PATH if save_res else None
if y_prob.is_cuda is True:
    y_prob = y_prob.to(torch.device('cpu'))
Metrics.evaluate_classification_binary(y_prob.numpy(), all_data.y_test_, saving_dir)


# saving model
if save_res is True:
    filename = join(RESULT_PATH, 'model.pt')
    torch.save(net, filename) # save all parameters, feature names must be included, not generate again.
# """
# explain model
interpret_model.run()

