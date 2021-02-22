from keras.models import Sequential,  Model
from keras.layers import Convolution2D,Input,BatchNormalization,Conv2D,Activation,Lambda,Subtract,Conv2DTranspose, PReLU
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
#from scipy.misc import imresize

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
def psnr(target, ref):
    # assume RGB image
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)

def interpolation(noisy , SNR , Number_of_pilot , interp):
    noisy_image = np.zeros((40000,72,14,2))

    noisy_image[:,:,:,0] = np.real(noisy)
    noisy_image[:,:,:,1] = np.imag(noisy)


    if (Number_of_pilot == 48):
        idx = [14*i for i in range(1, 72,6)]+[4+14*(i) for i in range(4, 72,6)]+[7+14*(i) for i in range(1, 72,6)]+[11+14*(i) for i in range(4, 72,6)]
    elif (Number_of_pilot == 16):
        idx= [4+14*(i) for i in range(1, 72,9)]+[9+14*(i) for i in range(4, 72,9)]
    elif (Number_of_pilot == 24):
        idx = [14*i for i in range(1,72,9)]+ [6+14*i for i in range(4,72,9)]+ [11+14*i for i in range(1,72,9)]
    elif (Number_of_pilot == 8):
      idx = [4+14*(i) for  i in range(5,72,18)]+[9+14*(i) for i in range(8,72,18)]
    elif (Number_of_pilot == 36):
      idx = [14*(i) for  i in range(1,72,6)]+[6+14*(i) for i in range(4,72,6)] + [11+14*i for i in range(1,72,6)]



    r = [x//14 for x in idx]
    c = [x%14 for x in idx]



    interp_noisy = np.zeros((40000,72,14,2))

    for i in range(len(noisy)):
        z = [noisy_image[i,j,k,0] for j,k in zip(r,c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,:,:,0] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,0] = z_intp
        z = [noisy_image[i,j,k,1] for j,k in zip(r,c)]
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,:,:,1] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,1] = z_intp


    interp_noisy = np.concatenate((interp_noisy[:,:,:,0], interp_noisy[:,:,:,1]), axis=0).reshape(80000, 72, 14, 1)
   
    
    return interp_noisy

def SRCNN_model():

    input_shape = (72,14,1)
    x = Input(shape = input_shape)
    c1 = Convolution2D( 64 , 9 , 9 , activation = 'relu', init = 'he_normal', border_mode='same')(x)
    c2 = Convolution2D( 32 , 1 , 1 , activation = 'relu', init = 'he_normal', border_mode='same')(c1)
    c3 = Convolution2D( 1 , 5 , 5 , init = 'he_normal', border_mode='same')(c2)
    #c4 = Input(shape = input_shape)(c3)
    model = Model(input = x, output = c3)
    ##compile
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error']) 
    return model
    
def SRCNN_train(train_data ,train_label, val_data , val_label , channel_model , num_pilots , SNR ):
    srcnn_model = SRCNN_model()
    print(srcnn_model.summary())
    
    checkpoint = ModelCheckpoint("./tmp/weights_{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=False,save_weights_only=True, mode='auto',period=10)
    callbacks_list = [checkpoint]

    srcnn_model.fit(train_data, train_label, batch_size=128, validation_data=(val_data, val_label),callbacks=callbacks_list, shuffle=True, epochs= 300 , verbose=0)
    
    #srcnn_model.save_weights("drive/codes/my_srcnn/SRCNN_SUI5_weights/SRCNN_48_12.h5")
    srcnn_model.save_weights("SRCNN_" + channel_model +"_"+ str(num_pilots) + "_"  + str(SNR) + ".h5")
   


def SRCNN_predict(input_data , channel_model , num_pilots , SNR):
    srcnn_model = SRCNN_model()
    srcnn_model.load_weights("SRCNN_" + channel_model +"_"+ str(num_pilots) + "_"  + str(SNR) + ".h5")
    predicted  = srcnn_model.predict(input_data)
    return predicted
def errors(y_true, y_pred):
    myOtherTensor = K.not_equal(y_true, K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))
def DNCNN_model (args):
    input_shape = (1,args.block_len, args.code_rate_k)
    inpt = Input(shape=input_shape)
    # 1st layer, Conv+relu
    x = Conv2D(filters=1, kernel_size=(args.kernel_size,args.kernel_size), strides=(1,1), data_format='channels_first',padding='same')(inpt)
    x = Activation('relu')(x)
    # 18 layers, Conv+BN+relu
    for i in range(args.num_layer):
        x = Conv2D(filters=64, kernel_size=(args.kernel_size,args.kernel_size), strides=(1,1), data_format='channels_first',padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(args.kernel_size,args.kernel_size), strides=(1,1), data_format='channels_first',padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=[errors])  
    # model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_crossentropy'])     
    
    return model

def DNCNN_train(args,train_data ,train_label, val_data , val_label,channel_model,identity):
  
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
  
  
