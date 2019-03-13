from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
from tensorflow.python import keras as K
from tensorflow.python.keras.callbacks import LearningRateScheduler, ModelCheckpoint
# import BatchNormalization
#from keras.layers.normalization import BatchNormalization

import tensorflow as tf

def _convlstm2d():
	###- Define and train model
	input_layer = Input(shape=[4, 240,320,3])
	output_layer = ConvLSTM2D(filters=16, kernel_size=(3, 3), strides=(1, 1),
	                   activation='relu', input_shape=(None, 4, 240, 320, 3),
	                   padding='same', return_sequences=True)(input_layer)
	return Model(input_layer, output_layer)


def _3dconv():
	###- Define and train model
	input_layer = Input(shape=[4, 240,320,3])
	output_layer = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
	               activation='relu',
	               padding='same', data_format='channels_last')(input_layer) 

	print('Creating basic CNN based LSTM...\n')

	return Model(input_layer, output_layer)


##- set GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

print(K.__version__)

print('Creating convlstm2d...\n')
model = _convlstm2d()
model.summary()
print('Creating 3dconv...\n')
model = _3dconv()
model.summary()

'''
RESULTS as fllows:
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 4, 240, 320, 3)    0         
_________________________________________________________________
conv_lst_m2d_1 (ConvLSTM2D)  (None, 4, 240, 320, 16)   11008     
=================================================================
Total params: 11,008
Trainable params: 11,008
Non-trainable params: 0
_________________________________________________________________
Creating 3dconv...

Creating basic CNN based LSTM...

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 4, 240, 320, 3)    0         
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 4, 240, 320, 16)   1312      
=================================================================
Total params: 1,312
Trainable params: 1,312
Non-trainable params: 0
_________________________________________________________________

'''


