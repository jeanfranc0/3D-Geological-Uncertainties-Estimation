from tensorflow.keras.layers import Add, LayerNormalization, BatchNormalization, concatenate, Flatten, AveragePooling1D, Reshape, Conv1D, Dense, MaxPool1D, Dropout, MaxPooling1D, ConvLSTM2D, ReLU, LSTM, GlobalMaxPooling1D, MultiHeadAttention, Conv3D, MaxPooling3D,Lambda, AveragePooling3D, GlobalMaxPooling3D, UpSampling3D, Concatenate, Cropping3D, ZeroPadding3D,Conv3DTranspose, TimeDistributed
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.backend import sqrt, mean, square, sum, epsilon
from tensorflow.keras.optimizers import Adam
from keras.layers.core import Activation
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
from keras import metrics
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

class ANN:
    def __init__(self, inputsXPor3D=(1, 1, 1, 1), inputsXRtp3D=(1, 1, 1, 1), inputsXNtg3D=(1, 1, 1, 1), inputsXPermi3D=(1, 1, 1, 1), inputXfluidUncertainties=(1, 1), n_outputs= (1), architecture='', pre_trained_model = None):
        
        print('init')
        seed_value = 42
        #tf.random.set_seed(seed_value)
        # Hyperparameters
        self.kernel_size = 3
        self.strides_size = 1
        self.dropout = 0.5
        self.pool_size_k = 3
        self.cnnArchitecture = architecture
        self.pre_trained_model = pre_trained_model
        self.architecture = {'resnet18': [3, 2], 'resnet34': [3, 3], 'resnet50': [3, 2], 'resnet101': [3, 2], 'seresnet18': [3, 2], 'seresnet34': [3, 2], 'seresnet50': [3, 2], 'seresnet101': [3, 2], 'mobilenetv2': [3, None], 'efficientnetv2-s': [3, 2], 'efficientnetv2-b0':[3, 2]}
        
        self.model = self.__build_model(inputsXPor3D,
                                        inputsXRtp3D,
                                        inputsXNtg3D, 
                                        inputsXPermi3D, 
                                        inputXfluidUncertainties,
                                        n_outputs)
    
    
    def __build_model(self, inputsXPor3D, inputsXRtp3D, inputsXNtg3D, inputsXPermi3D, inputXfluidUncertainties, n_outputs):

        print('build model')
        epsilon_std = 1.0
        Por3D = Input(shape=inputsXPor3D, name='Por3D_input')
        Rtp3D = Input(shape=inputsXRtp3D, name='Rtp3D_input')
        Ntg3D = Input(shape=inputsXNtg3D, name='Ntg3D_input')
        Permi3D = Input(shape=inputsXPermi3D, name='Permi3D_input')
        fluidUncertainties = Input(shape=inputXfluidUncertainties, name='fluidUncertainties_input')
        
        # Encode Porosity
        e1 = Reshape((Por3D.shape[1] * Por3D.shape[2] * Por3D.shape[3],))(Por3D)
        
        # Encode Rock type
        e2 = Reshape((Rtp3D.shape[1] * Rtp3D.shape[2] * Rtp3D.shape[3],))(Rtp3D)
        
        # Encode NTG
        e3 = Reshape((Ntg3D.shape[1] * Ntg3D.shape[2] * Ntg3D.shape[3],))(Ntg3D)
        
        # Encode Permiability I
        e4 = Reshape((Permi3D.shape[1] * Permi3D.shape[2] * Permi3D.shape[3],))(Permi3D)
        
        # Scalar properties
        x7 = Reshape((fluidUncertainties.shape[1],))(fluidUncertainties)
        
        # Regression
        reg = concatenate([e1, e2, e3, e4, x7])
        reg = Dense(units=128,
                    kernel_regularizer=regularizers.L2(1e-5),
                    bias_regularizer=regularizers.L2(1e-5),
                    activity_regularizer=regularizers.L2(1e-5), 
                    activation ='relu')(reg)
        reg = Dropout(self.dropout)(reg)
        reg = Dense(units=64,
                    kernel_regularizer=regularizers.L2(1e-5),
                    bias_regularizer=regularizers.L2(1e-5),
                    activity_regularizer=regularizers.L2(1e-5), 
                    activation ='relu')(reg)
        reg = Dropout(self.dropout)(reg)
        reg = Dense(units=n_outputs,
                    kernel_regularizer=regularizers.L2(1e-5),
                    bias_regularizer=regularizers.L2(1e-5),
                    activity_regularizer=regularizers.L2(1e-5),
                    activation ='sigmoid')(reg)
        
        def rmse(y_true, y_pred):
            return sqrt(mean(square(K.flatten(y_pred) - K.flatten(y_true))))

        def r2(y_true, y_pred):
            return 1 - K.sum(K.square(y_true - y_pred)) / K.sum(K.square(y_true - K.mean(y_true)))

        def correlation(x, y):    
            mx = tf.math.reduce_mean(x)
            my = tf.math.reduce_mean(y)
            xm, ym = x-mx, y-my
            r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
            r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
            return r_num / r_den

        def smape_loss(y_true, y_pred):
            y_true = K.flatten(y_true)
            y_pred = K.flatten(y_pred)
            epsilon = 0.1
            summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
            smape = K.abs(y_pred - y_true) / summ * 2.0
            return smape
        
        vae = Model([Por3D, Rtp3D, Ntg3D, Permi3D, fluidUncertainties], 
                    reg)
        
        # Define the VAE loss function
        def vae_loss(Por3D,reconstructionPOR3D):
            dimOriginal = 24 * 38 * 73
            loss_reconPor3D = mse(K.flatten(Por3D), K.flatten(reconstructionPOR3D))
            loss_reconRtp3D = mse(K.flatten(Rtp3D), K.flatten(reconstructionRTP3D))
            loss_reconNtg3D = mse(K.flatten(Ntg3D), K.flatten(reconstructionNTG3D))
            loss_reconPermi3D = mse(K.flatten(Permi3D), K.flatten(reconstructionPERM3D))
            loss_KL = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(loss_KL + loss_reconPor3D + loss_reconRtp3D + loss_reconNtg3D + loss_reconPermi3D)

        vae.compile(loss=smape_loss,
                    metrics = ['mae',rmse,smape_loss, r2, correlation],
                    optimizer = 'adam')
        vae.summary()
        import numpy as np
        import math
        import sklearn
        from scipy.stats import linregress
        from sklearn.metrics import r2_score

        def metrics(true_values, predicted_values):
            slope, intercept, r_value, p_value, std_err = linregress(true_values.flatten(), predicted_values.flatten())
            #r2 = r2_score(true_values.flatten(), predicted_values.flatten())
            print('r_value', round(r_value,3))
        def imageNormalization(image):
            return (image-np.min(image))/(np.max(image)-np.min(image))  

        return vae
