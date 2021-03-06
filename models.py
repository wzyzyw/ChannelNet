from keras.models import Sequential,  Model
from keras.layers import Convolution2D,Input,BatchNormalization,Conv2D,Activation,Lambda,Subtract,Conv2DTranspose, PReLU,Conv1D
from keras.regularizers import l2
from keras.layers import  Reshape,Dense,Flatten
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from scipy.io import loadmat
import keras.backend as K
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy as np
import math
from scipy import interpolate
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
#from scipy.misc import imresize
def enhancedloss(y_true,y_pred):
    lamb=0.01
    num=6
    r=y_pred-y_true
    r_ave=K.mean(r)
    S=K.mean((r-r_ave)**3)/(K.mean((r-r_ave)**2)**(3/2))
    C=K.mean((r-r_ave)**4)/(K.mean((r-r_ave)**2)**2)
    normal_test=S**2/(6/num)+(C-3)**2/(24/num)
    return K.mean((y_pred-y_true)**2)+lamb*normal_test
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_loss = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.train_loss['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.train_loss['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type):
        iters = range(len(self.train_loss[loss_type]))
        plt.figure()
        # loss
        plt.semilogy(iters, self.train_loss[loss_type], 'blue', label='train loss')
        if loss_type == 'epoch':
            # val_loss
            plt.semilogy(iters, self.val_loss[loss_type], 'y', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def errors(y_true, y_pred):
    myOtherTensor = K.not_equal(y_true, K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))
def DNCNN_model (args):
    if args.use_normaltest:
        myloss=enhancedloss
    else:
        myloss='mean_squared_error' 
    newlen=args.block_len+args.remainlen
    newn=args.code_rate_n+args.remainn
    input_shape = (newlen, newn+1 if args.use_noisemap else newn)
    inpt = Input(shape=input_shape)
    # 1st layer, Conv+relu
    x = Conv1D(filters=64, kernel_size=args.kernel_size, strides=1,padding='same')(inpt)
    x = Activation('relu')(x)
    # 18 layers, Conv+BN+relu
    for i in range(args.num_layer):
        x = Conv1D(filters=64, kernel_size=args.kernel_size, strides=1, padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv1D(filters=newn, kernel_size=args.kernel_size, strides=1, padding='same')(x)
    # x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    model.compile(optimizer=adam, loss=myloss, metrics=[myloss])  
    # model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_crossentropy'])    
    
    return model

def DNCNN_train(args,train_data ,train_label, val_data , val_label,channel_model,identity):
  
    config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=False)
    config.gpu_options.allow_growth=True

    # frac = args.GPU_proportion
    # config.gpu_options.per_process_gpu_memory_fraction = frac
    set_session(tf.Session(config=config))

    dncnn_model = DNCNN_model(args)
    print(dncnn_model.summary())

    losslog=LossHistory()
    checkpoint = ModelCheckpoint("./tmp/weights_{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=False,save_weights_only=True, mode='auto',period=10)
    callbacks_list = [checkpoint,losslog]

    dncnn_model.fit(train_data, train_label, batch_size=args.batch_size, validation_data=(val_data, val_label),callbacks=callbacks_list, shuffle=True, epochs= args.num_epoch , verbose=2)
    dncnn_model.save_weights("./tmp/DNCNN_" + channel_model +"_"+str(identity)+ ".h5")
    losslog.loss_plot('epoch')
  
  
  
def DNCNN_predict(args,input_data, channel_model ,identity ):
    dncnn_model = DNCNN_model(args)
    dncnn_model.load_weights("./tmp/DNCNN_" + channel_model +"_"+ str(identity)  + ".h5")
    predicted  = dncnn_model.predict(input_data)
    return predicted
  
  
