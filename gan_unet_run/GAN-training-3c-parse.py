
# coding: utf-8

# In[ ]:


'''
made by weiyw @ 2019-4-10
GAN
'''
## network workflow
import os
import keras
import segyio
import argparse
import datetime
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv2DTranspose
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Lambda,
    Reshape,
    Dropout,
    Concatenate
)
os.environ["CUDA_VISIBLE_DEVICES"]="6"


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
        parser.add_argument('-sample_interval', type=int, default=10, help='save per epoch')
        parser.add_argument('-nph', type=int, default=3, help='how many phase in one data')
        parser.add_argument('-nop', type=int, default=1, help='how many phase in one data')
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


class GANDataloader():
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
#         i = 0
        with segyio.open(self.data_path,'r',ignore_geometry=True) as segyfile: 
            segyfile.mmap()
#             while True:
            for i in range(self.n_batches):
#                 if (i + 1) * batch_size > 151 * ratio:
#                     i = 0
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
#                 i = i + 1


# In[ ]:


class vspGAN():
    def __init__(self, nt, nr, nph, nop):
        self.gf = 64
        self.df = 64
        self.nt = nt#2001
        self.nr = nr#467
        self.nph = nph#3
        self.nop = nop#1
        
        self.disc_patch = (int(self.nt / 2**4)+1, int(self.nr / 2**4)+1, 1)
        optimizer = Adam(0.0002, 0.5)
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])
        
        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        data_B = Input( shape=( self.nt, self.nr, self.nph) )
        data_A = Input( shape=( self.nt, self.nr, self.nop) )

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(data_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([data_B, fake_A])

        self.combined = Model(inputs=[data_B, data_A], outputs=[fake_A,valid])
        self.combined.compile(loss=['mse', 'mae'],loss_weights=[1, 100],optimizer=optimizer) 
        
    def build_generator(self):  
        '''
        U-net generator
        '''
        def cutLayer(xx, target):
            return xx[:, 0:int(target.shape[1]),0:int(target.shape[2]),0:int(target.shape[3])]
    
        def conv2d(layer_input, filters, f_size=(4,4), s_size=(2,2), bn=True):
            """Layers used during downsampling"""
            xx = Conv2D(filters=filters, kernel_size=f_size, strides=s_size, padding='same', 
                        data_format='channels_last', dilation_rate=(1, 1), activation=None, 
                        use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros', 
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                        kernel_constraint=None, bias_constraint=None)(layer_input)
            if bn:
                xx = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                           beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', 
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 
                           beta_constraint=None, gamma_constraint=None)(xx)
            return xx
        

        def deconv2d(layer_input, skip_input, filters, f_size=(4,4),s_size=(2,2), dropout_rate=0, if_skip=True,
                     if_last=False):
            """Layers used during upsampling"""
            xx = Conv2DTranspose(filters=filters, kernel_size=f_size, strides=s_size, padding='same', output_padding=None, 
                         data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, 
                         kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                         kernel_constraint=None, bias_constraint=None)(layer_input)
            if if_last:
                xx = Conv2DTranspose(filters=filters, kernel_size=f_size, strides=s_size, padding='same', output_padding=None, 
                             data_format='channels_last', dilation_rate=(1, 1), activation='tanh', use_bias=True, 
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)(layer_input)
                
            if if_skip and xx.shape != skip_input.shape:
                xx = Lambda(cutLayer, arguments={'target':(skip_input)})(xx)#(xx,skip_input)
                
            if dropout_rate:   
                xx = Dropout(dropout_rate)(xx)
                
            if not if_last:
                xx = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                           beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', 
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 
                           beta_constraint=None, gamma_constraint=None)(xx)
                xx = Concatenate()([xx, skip_input])
            return xx

        # Image input
        d0 = Input( shape=( self.nt, self.nr, self.nph) )#shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*8)
        d3 = conv2d(d2, self.gf*16)
        d4 = conv2d(d2, self.gf*16)

        # Upsampling
    
        u1 = deconv2d(d4, d3, self.gf*16)
        u2 = deconv2d(u1, d2, self.gf*16)
        u3 = deconv2d(u2, d1, self.gf*8)
        u4 = deconv2d(u3, d0, self.nop, if_last=True)

        return Model(inputs = d0, outputs = u4)
    
    
    def build_discriminator(self):  
        '''
        U-net discriminator
        ''' 
        def d_layer(layer_input, filters, f_size=(4,4),s_size=(2,2), bn=True):
            
            d = Conv2D(filters, kernel_size=f_size, strides=s_size, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        data_B = Input( shape=( self.nt, self.nr, self.nph) )
        data_A = Input( shape=( self.nt, self.nr, self.nop) )
        
        # Concatenate image and conditioning image by channels to produce input
        combined_data = Concatenate(axis=-1)([data_B, data_A])

        d1 = d_layer(combined_data, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([data_B, data_A], validity)
        


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


opt = parameters().parse()
GANmodel = vspGAN(opt.nt, opt.nr, opt.nph, opt.nop)
D = GANmodel.discriminator
G = GANmodel.generator
C = GANmodel.combined
my_data_loader = GANDataloader(opt.data_path, opt.nt, opt.nr, opt.nph)


# In[ ]:


disc_patch = GANmodel.disc_patch
valid = np.ones((opt.batch_size,) + disc_patch )
fake = np.zeros((opt.batch_size,) + disc_patch ) 

start_time = datetime.datetime.now()
save_dir = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
for epoch in range(opt.epoch):
    for batch_i, (data_B, data_A) in enumerate(
        my_data_loader.load_batch(batch_size=opt.batch_size, is_testing=False, ratio=opt.ratio)):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Condition on B and generate a translated version
        fake_A = G.predict(data_B)
        
        # Train the discriminators (original images = real / generated = Fake)
        d_loss_real = D.train_on_batch([data_B, data_A], valid)
        d_loss_fake = D.train_on_batch([data_B, fake_A], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # -----------------
        #  Train Generator
        # -----------------

        # Train the generators
        g_loss = C.train_on_batch([data_B, data_A], [data_A, valid])
        
        elapsed_time = datetime.datetime.now() - start_time
        # Plot the progress
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % 
               (epoch+1, opt.epoch, batch_i+1, my_data_loader.n_batches,
                d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time))
    if epoch % opt.sample_interval == 0:
        this_name = opt.out_name + "-" + str(epoch+1)
        D.save_weights(os.path.join( save_dir, '{}-D.h5'.format(this_name)), overwrite=True)
        G.save_weights(os.path.join( save_dir, '{}-C.h5'.format(this_name)), overwrite=True)
        print("model saved")
print("training finished")

