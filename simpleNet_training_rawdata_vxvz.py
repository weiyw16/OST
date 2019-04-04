
# coding: utf-8

# In[ ]:


## headings
"""
made by weiyw @ 2019-04-04
made to use both vz and vx components
"""
import os
import time
import struct
import argparse
import numpy as np
from glob import glob

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import keras
from keras.models import Model
from keras.layers import Conv2D#Convolution2D
# from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers import (Input, Activation, merge, Dense, Reshape)
import metrics as metrics


os.environ["CUDA_VISIBLE_DEVICES"]="0" # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)


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


def get_min_nonzero_value(data_in, init_min = 1):
    min_value = init_min
    for bb in data_in:
        for aa in bb:
            if abs(aa) < min_value and aa != 0:
                min_value = abs(aa)
    return min_value


# In[ ]:


def get_bindata(data_path, data_index, data_phase, nt, nr):
    # data_index start from 1
    model_num = int( ( int(data_index) - 1 ) / 51) + 1
    source_num = ( int(data_index) - 1 ) % 51 + 1
    data_name = data_path + "/model" + str(model_num) + "source" + str(source_num) + data_phase + ".bin"

    out_data = np.empty((nt,nr))
    FA = open(data_name, "rb")
    FA.seek(3232,0)
    for tt in range(nt):
        for rr in range(nr):
            data = FA.read(4)
            data_float = struct.unpack("f", data)[0]
            out_data[tt][rr] = data_float
    return out_data  


# In[ ]:


class Dataloader():
    def __init__(self, data_path, nt, nr, nph):
        self.data_path = data_path
        self.nt = nt
        self.nr = nr
        self.nph = nph
    
    def readin_bin_data(self, data_name):
        out_data = np.empty((self.nt, self.nr))
        FA = open(data_name, "rb")
        FA.seek(3232,0)
        for tt in range(self.nt):
            for rr in range(self.nr):
                data = FA.read(4)
                data_float = struct.unpack("f", data)[0]
                out_data[tt][rr] = data_float
        return out_data      
    
    def load_batch(self, batch_size=1, is_testing=False, ratio=0.5):
        data_type = "train" if not is_testing else "test"
        path = glob('%s/vz/*' % self.data_path)
       
        self.n_batches = int( len(path) / batch_size * ratio )
        for i in range(self.n_batches):
            x_data = np.empty((batch_size, self.nt, self.nr, self.nph)) ## b-4000-300-2, vz, vx
            y_data = np.empty((batch_size, self.nt, self.nr, self.nph)) ## b-4000-300-2, div, curl
            for data_index_local in range(batch_size): ## 0 ~ batchsize-1
                data_index = i * batch_size + data_index_local + 1
                model_num = int( ( int(data_index) - 1 ) / 51) + 1
                source_num = ( int(data_index) - 1 ) % 51 + 1
                if self.nph == 2:
                    for data_phase in ['vz', 'vx']:
                        p_flag = int(data_phase == 'vx')
                        data_name = self.data_path + "/" + data_phase +                         "/model" + str(model_num) + "source" + str(source_num) + data_phase + ".bin"
                        x_data[data_index_local, :, :, p_flag] = self.readin_bin_data(data_name)
                    for data_phase in ['div', 'curl']:
                        p_flag = int(data_phase == 'curl')
                        data_name = self.data_path + "/" + data_phase +                         "/model" + str(model_num) + "source" + str(source_num) + data_phase + ".bin"
                        y_data[data_index_local, :, :, p_flag] = self.readin_bin_data(data_name)
                if self.nph == 1:
                    data_phase = 'vz'
                    data_name = self.data_path + "/" + data_phase +                     "/model" + str(model_num) + "source" + str(source_num) + data_phase + ".bin"
                    x_data[data_index_local, :, :, 0] = self.readin_bin_data(data_name)
                    data_phase = 'div'
                    data_name = self.data_path + "/" + data_phase +                     "/model" + str(model_num) + "source" + str(source_num) + data_phase + ".bin"
                    y_data[data_index_local, :, :, 0] = self.readin_bin_data(data_name)                    
                yield x_data, y_data
                


# In[ ]:


def simpleNet(nt, nr, nph):
    x_input = Input( shape=( nt, nr, nph) )##one_piece
    conv1 = Conv2D(
        nb_filter=64, nb_row=3, nb_col=3, border_mode="same", data_format="channels_last")(x_input)
    conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                               beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', 
                               moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 
                               beta_constraint=None, gamma_constraint=None)(conv1)
    conv2 = Conv2D(
        nb_filter=2, nb_row=3, nb_col=3, border_mode="same", data_format="channels_last")(conv1)

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


# data_path = "/media/wywdisk/VSPdata/data/haveinvx/layer2_haveinvx"

# nt = 4000     # time step
# nr = 400      # receiver
# ns = 51       # shot
# nmodel = 2    # batch
# nph = 2

# ## training parameters
# TPepoches = 4
# TPbatch_size = 4

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
callbacks = [checkpoint]#, lr_reducer, lr_scheduler]


# In[ ]:


## training
log = model.fit_generator(my_data_loader.load_batch(batch_size=opt.batch_size, is_testing=False, ratio=opt.ratio),              steps_per_epoch=opt.batch_size, epochs=opt.epoch, verbose=1, callbacks=callbacks, validation_data=None,               validation_steps=None, class_weight=None, max_queue_size=10,               workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)


# In[ ]:


## saving last model
json_string = model.to_json()
open('model_json', 'w').write(json_string)
model.save_weights(os.path.join( save_dir, '{}.final.best.h5'.format(opt.out_name)), overwrite=True)


# In[ ]:


# %hist -f simpleNet_training_rawdata_vxvz.py

