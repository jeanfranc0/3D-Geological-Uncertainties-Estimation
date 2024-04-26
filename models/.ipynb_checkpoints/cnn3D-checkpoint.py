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

class CNN3D:
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
        
        self.model = self.__build_model(inputsXPor3D,
                                        inputsXRtp3D,
                                        inputsXNtg3D, 
                                        inputsXPermi3D, 
                                        inputXfluidUncertainties,
                                        n_outputs)
    
    def encoderBlock(self, uncertainty):
        width = 2
        kernel_size = 3
        pool_size_k = 2
        dropout = 0.5
        strides_size= 5  
        reg_weights = 0.00001
        
        def conv_bn_relu(nb_filter, kernel_size, stride = None):
            def conv_func(x):
                if stride:
                    x1 = Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size),strides=stride,
                                bias_regularizer=regularizers.L2(reg_weights),
                                padding='same', 
                                kernel_regularizer=regularizers.L2(reg_weights))(x)
                else:
                    x1 = Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size),
                                bias_regularizer=regularizers.L2(reg_weights),
                                padding='same', 
                                kernel_regularizer=regularizers.L2(reg_weights))(x)
                x_bn = BatchNormalization()(x1)
                x_relu = Activation("relu")(x_bn)
                return x1, x_relu

            return conv_func
        
        # CONV3D
        encod1, encod1_relu = conv_bn_relu(width ** 4, kernel_size, stride=2)(uncertainty)
        encod1_relu = Dropout(self.dropout)(encod1_relu)
        encod2, encod2_relu = conv_bn_relu(width ** 4, kernel_size)(encod1_relu)
        # POOL3D
        encod3 = MaxPooling3D(pool_size=3)(encod2_relu)
        encod3 = Dropout(dropout)(encod3)
        
        # 2 CONV3D
        encod11, encod11_relu = conv_bn_relu(width ** 5, kernel_size)(encod3)
        encod11_relu = Dropout(self.dropout)(encod11_relu)
        encod22, encod22_relu = conv_bn_relu(width ** 5, kernel_size)(encod11_relu)
        
        def residual_block(input_tensor, filters, kernel_size):
            x = Conv3D(filters, kernel_size, padding='same')(input_tensor)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv3D(filters, kernel_size, padding='same')(x)
            x = BatchNormalization()(x)
            x = Add()([input_tensor, x])
            x = ReLU()(x)
            return x
        
        # RESIDUAL
        encod3 = residual_block(encod22, width ** 5, (3, 3, 3))
        
        # 3 CONV3D
        encod4, encod4_relu = conv_bn_relu(width ** 6, kernel_size, stride=2)(encod3)
        encod4_relu = Dropout(self.dropout)(encod4_relu)
        encod5, encod5_relu = conv_bn_relu(width ** 6, kernel_size)(encod4_relu)
        # POOL3D
        encod6 = MaxPooling3D(pool_size=2)(encod5_relu)
        encod6 = Dropout(dropout)(encod6)
        
        encod7 = Flatten()(encod6)
        e = Dense(256,
                   kernel_regularizer=regularizers.L2(1e-4),
                   bias_regularizer=regularizers.L2(1e-4),
                   activity_regularizer=regularizers.L2(1e-5))(encod7)
                
                
        return e, e, e, e, e, encod5, encod4, encod2, encod1, encod1

      
    def __build_model(self, inputsXPor3D, inputsXRtp3D, inputsXNtg3D, inputsXPermi3D, inputXfluidUncertainties, n_outputs):

        print('build model')
        epsilon_std = 1.0
        Por3D = Input(shape=inputsXPor3D, name='Por3D_input')
        Rtp3D = Input(shape=inputsXRtp3D, name='Rtp3D_input')
        Ntg3D = Input(shape=inputsXNtg3D, name='Ntg3D_input')
        Permi3D = Input(shape=inputsXPermi3D, name='Permi3D_input')
        fluidUncertainties = Input(shape=inputXfluidUncertainties, name='fluidUncertainties_input')
        
        # Encode Porosity
        e1, Por3D_encod11, Por3D_encod10, Por3D_encod8, Por3D_encod7, Por3D_encod5, Por3D_encod4, Por3D_encod2, Por3D_encod1, Por3D_conv1 = self.encoderBlock(
            Por3D
        )
        
        # Encode Rock type
        e2, Rtp3D_encod11, Rtp3D_encod10, Rtp3D_encod8, Rtp3D_encod7, Rtp3D_encod5, Rtp3D_encod4, Rtp3D_encod2, Rtp3D_encod1, Rtp3D_conv1 = self.encoderBlock(
            Rtp3D
        )
        
        # Encode NTG
        e3, Ntg3D_encod11, Ntg3D_encod10, Ntg3D_encod8, Ntg3D_encod7, Ntg3D_encod5, Ntg3D_encod4, Ntg3D_encod2, Ntg3D_encod1, Ntg3D_conv1 = self.encoderBlock(
            Ntg3D
        )
        
        # Encode Permiability I
        e4, Permi3D_encod11, Permi3D_encod10, Permi3D_encod8, Permi3D_encod7, Permi3D_encod5, Permi3D_encod4, Permi3D_encod2, Permi3D_encod1, Permi3D_conv1 = self.encoderBlock(
            Permi3D
        )
        
        # Scalar properties
        x7 = Reshape((fluidUncertainties.shape[1],))(fluidUncertainties)
        
        # Regression
        reg = concatenate([e1, e2, e3, e4, x7])
        reg = Dense(units=512,
                       kernel_regularizer=regularizers.L2(1e-5),
                       bias_regularizer=regularizers.L2(1e-5),
                       activity_regularizer=regularizers.L2(1e-5), 
                       activation ='relu')(reg)
        reg = Dropout(self.dropout)(reg)
        reg = Dense(units=n_outputs,
                       kernel_regularizer=regularizers.L2(1e-5),
                       bias_regularizer=regularizers.L2(1e-5),
                       activity_regularizer=regularizers.L2(1e-5))(reg)
        
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
        
        vae.compile(loss=smape_loss,
                    metrics = ['mae',rmse,smape_loss, r2, correlation],
                    optimizer = 'adam')

        vae.summary()
        
        return vae
