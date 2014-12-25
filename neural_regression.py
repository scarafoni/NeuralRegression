"""runs iterations to train a neural network"""

import numpy as np
import cPickle as pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

train_file = 'data/train.csv'
validation_file = 'data/validation.csv'
hidden_size = 50
epochs = 5

# load data
train = np.loadtxt( train_file, delimiter = ',' )
validation = np.loadtxt( validation_file, delimiter = ',' )
train = np.vstack(( train, validation ))
x_train = train[:,0:-1]
print('x_train', x_train)
y_train = train[:,-1]
print('y_train', y_train)
y_train = y_train.reshape( -1, 1 )
print('y_train2', y_train)
input_size = x_train.shape[1]
print('input_size', input_size)
target_size = y_train.shape[1]
print('target_size', target_size)

# prepare dataset
ds = SDS( input_size, target_size )
ds.setField( 'input', x_train )
ds.setField( 'target', y_train )

# init and train
net = buildNetwork( input_size, hidden_size, target_size, bias = True )
trainer = BackpropTrainer( net,ds )
print "training for {} epochs...".format( epochs )
for i in range( epochs ):
    mse = trainer.train()
    rmse = sqrt( mse )
    print "training RMSE, epoch {}: {}".format( i + 1, rmse )
