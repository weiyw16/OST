
# coding: utf-8

# In[ ]:


## headings
"""
made by weiyw @ 2019-04-07
made to use both three components in segy data
"""
import os
import time
import struct
import segyio
import argparse
import numpy as np

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import keras
from keras.models import Model
from keras.layers import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import (Input, Activation, merge, Dense, Reshape)
import metrics as metrics

os.environ["CUDA_VISIBLE_DEVICES"]="0" 


# In[ ]:


class parameters():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        parser.add_argument('-data_path', required=True, help='path of readin data')
        parser.add_argument('-out_name', required=True, help='the name of output model')
        parser.add_argument('-nt', type=int, default=4000, help='time steps')
        parser.add_argument('-nr', type=int, default=400, help='receivers')
        parser.add_argument('-ns', type=int, default=51, help='time steps')
        parser.add_argument('-batch_size', type=int, default=2, help='batch size')
        parser.add_argument('-nph', type=int, default=2, help='how many phase in one data, 2 for vx vz; 1 for vz')
        parser.add_argument('-epoch', type=int, default=100, help='epoch')
        parser.add_argument('-ratio', type=float, default=0.5, help='the ratio of training data in the whole dataset')
        self.initialized = True
        return parser
    
    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()
    
    def parse(self):
        opt = self.gather_options()
        self.opt = opt
        return self.opt
    


# In[ ]:


class Dataloader():
    def __init__(self, data_path, nt, nr, nph):
        self.data_path = data_path
        self.nt = nt
        self.nr = nr
        self.nph = nph

    def normaliza(self, datain):
        for batch_ii in range( datain.shape[0] ):
            for iterm_ii in range( datain.shape[-1] ):
                target = datain[batch_ii, :, :, iterm_ii]
                mind = target.min() #datain[batch_ii, :, :, iterm_ii].min()
                maxd = target.max() #datain[batch_ii, :, :, iterm_ii].max()      
                for ii in range(target.shape[0]):
                    for jj in range(target.shape[1]):
                        if target[ii, jj] > 0: 
                            target[ii,jj] = target[ii,jj] / maxd
                        if target[ii, jj] < 0:
                            target[ii,jj] = target[ii,jj] / ( 0 - mind )
                datain[batch_ii, :, :, iterm_ii] = target
    #     return datain
    
    def load_batch(self, batch_size=1, is_testing=False, ratio=0.5):        
        self.n_batches = int( 151 / batch_size * ratio ) #int( len(path) / batch_size * ratio )
        x_data = np.empty((batch_size, self.nt, self.nr, self.nph)) ## b-2001-467-4, acc
        y_data = np.empty((batch_size, self.nt, self.nr, 1)) ## b-2001-467-1, div, curl
        i = 0
        with segyio.open(self.data_path,'r',ignore_geometry=True) as segyfile: 
            segyfile.mmap()
            while True:
#             for i in range(self.n_batches):
                if (i + 1) * batch_size > 151 * ratio:
                    i = 0
#                     break
                for batch_i in range(batch_size):
                    for nr_i in range(self.nr):
#                     with segyio.open(self.data_path,'r',ignore_geometry=True) as segyfile:      
                        y_data[batch_i,:,nr_i,0] =                         segyfile.trace[i*batch_size*4*self.nr + batch_i * 4 * self.nr + nr_i * 4 + 0]
                        x_data[batch_i,:,nr_i,0] =                         segyfile.trace[i*batch_size*4*self.nr + batch_i * 4 * self.nr + nr_i * 4 + 1]
                        x_data[batch_i,:,nr_i,1] =                         segyfile.trace[i*batch_size*4*self.nr + batch_i * 4 * self.nr + nr_i * 4 + 2]
                        x_data[batch_i,:,nr_i,2] =                         segyfile.trace[i*batch_size*4*self.nr + batch_i * 4 * self.nr + nr_i * 4 + 3]                 
                self.normaliza(x_data)
                self.normaliza(y_data)
                yield x_data, y_data
                i = i + 1


# In[ ]:


def simpleNet(nt, nr, nph):
    x_input = Input( shape=( nt, nr, nph) )##one_piece
#     conv1 = Conv2D(
#         nb_filter=64, nb_row=3, nb_col=3, padding="same", data_format="channels_last")(x_input)
    conv1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last', 
                   dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                   bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x_input)
    conv1 = LeakyReLU()(conv1)
    conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                               beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', 
                               moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 
                               beta_constraint=None, gamma_constraint=None)(conv1)
#     conv1_1 = Conv2D(
#         nb_filter=128, nb_row=3, nb_col=3, padding="same", data_format="channels_last")(conv1)
    conv1_1 = Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last', 
                   dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                   bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(conv1)
    conv1_1 = LeakyReLU()(conv1_1)
    conv1_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                               beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', 
                               moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 
                               beta_constraint=None, gamma_constraint=None)(conv1_1)
    
    conv1_2 = Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last', 
                   dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                   bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(conv1_1)
    conv1_2 = LeakyReLU()(conv1_2)
#     conv1_2 = Conv2D(
#         nb_filter=128, nb_row=3, nb_col=3, padding="same", data_format="channels_last")(conv1_1)
    conv1_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                               beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', 
                               moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 
                               beta_constraint=None, gamma_constraint=None)(conv1_2)
#     conv1_3 = Conv2D(
#         nb_filter=128, nb_row=3, nb_col=3, padding="same", data_format="channels_last")(conv1_2)
    conv1_3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last', 
                   dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                   bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(conv1_2) 
    conv1_3 = LeakyReLU()(conv1_3)
    conv1_3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                               beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', 
                               moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 
                               beta_constraint=None, gamma_constraint=None)(conv1_3)
    
    conv2 = Conv2D(filters=1, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last', 
                   dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                   bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(conv1_3)


#     conv2 = Conv2D(
#         nb_filter=1, nb_row=3, nb_col=3, padding="same", data_format="channels_last")(conv1_3)

    model = Model(inputs=x_input, outputs=conv2)
    return model


# In[ ]:


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


# In[ ]:


## setup
opt = parameters().parse()
model = simpleNet(opt.nt, opt.nr, opt.nph)
model.summary()
my_data_loader = Dataloader(opt.data_path, opt.nt, opt.nr, opt.nph)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])


# In[ ]:


## saveing parameters
save_dir = os.path.join(os.getcwd(), 'saved_models')

filepath = os.path.join(save_dir,'{}.best.h5'.format(opt.out_name))
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=0, save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
callbacks = [checkpoint] #, lr_reducer, lr_scheduler]


# In[ ]:


## training
log = model.fit_generator(my_data_loader.load_batch(batch_size=opt.batch_size, is_testing=False, ratio=opt.ratio),              steps_per_epoch=int( (151*opt.ratio-opt.batch_size) /opt.batch_size ), epochs=opt.epoch, verbose=1, callbacks=callbacks, validation_data=None,               validation_steps=None, class_weight=None, max_queue_size=10,               workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)


# In[ ]:


## saving last model
json_string = model.to_json()
open('model_json', 'w').write(json_string)
model.save_weights(os.path.join( save_dir, '{}.final.best.h5'.format(opt.out_name)), overwrite=True)


# In[ ]:


# %hist -f simpleNet_training_rawdata_vxvz.py

