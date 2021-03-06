"""runs iterations to train a neural network"""

import numpy as np
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

train_file = 'train.csv'
validation_file = 'validation.csv'
hidden_size = 5
epochs = 5

# load data
train = np.loadtxt( train_file, delimiter = ',' )
validation = np.loadtxt( validation_file, delimiter = ',' )
train = np.vstack((train, validation))
x_train = train[:,0:-1]
y_train = train[:,-1]
y_train = y_train.reshape(-1, 1)
input_size = x_train.shape[1]
target_size = y_train.shape[1]

# prepare dataset
ds = SDS(input_size, target_size) 
ds.setField('input', x_train)
ds.setField('target', y_train)

# init and train
net = buildNetwork(input_size, hidden_size, target_size, bias=True)
trainer = BackpropTrainer(net, ds)
print "training for {} epochs".format(epochs)
for i in range(epochs):
    mse = trainer.train()
    rmse = sqrt(mse)
    print "training RMSE, epoch {}: {}".format(i+1, rmse)
