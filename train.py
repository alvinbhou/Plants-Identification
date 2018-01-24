import numpy as np
import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import regularizers
from keras.layers import Conv2D, BatchNormalization, Input, Activation
from keras.layers import LeakyReLU, Lambda, Reshape, Concatenate, Add, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.models import Model, load_model

log_name = 'resnet50_log.csv'
save_model_name = 'resnet50_model.h5'

train_path = 'data/train'
valid_path = 'data/valid'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), batch_size=32)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), batch_size=32)


def create_base_model( w = 'imagenet', trainable = False):
    model = ResNet50(weights=w, include_top=False, input_shape=(224, 224, 3))
    if(not trainable):
        for layer in model.layers:
            layer.trainable = False
    return model

def create_InceptionResNetV2_model( w = 'imagenet', trainable = False):
    model = InceptionResNetV2(weights=w, include_top=False, input_shape=(224, 224, 3))
    if(not trainable):
        for layer in model.layers:
            layer.trainable = False
    return model

def rebase_base_model(model):
    # for layer in model.layers[:self.num_fixed_layers]:
    #     layer.trainable = False
    for layer in model.layers:
        layer.trainable = True
    return model


def add_custom_layers(base_model):
    CLASSES = 240
    x = base_model.output
    x = Flatten()(x)
    # x = Dropout(0.2)(x)
    # x = Dense(1024, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
    x = Dense(4096, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
    # x = Dense(2048, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
    y = Dense(CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=y)
    return model

# Build Fine-tuned VGG16 model
def vgg16_model():
    vgg16_model = keras.applications.vgg16.VGG16()
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    model.layers.pop()

    for layer in model.layers:
        layer.trainable = False

    model.add(Dense(4096, activation='relu', name='fc3'))
    model.add(Dense(240, activation='softmax'))
    return model

sv = ModelCheckpoint('InceptionResNetV2_model.h5',
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='auto',
                          )

model = create_base_model(w = 'imagenet', trainable = False)
# model = create_InceptionResNetV2_model(w = 'imagenet', trainable = False)
model = add_custom_layers(model)

# parallel_model = multi_gpu_model(model, gpus=2)
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

csv_logger = CSVLogger(log_name, append=True, separator=';')
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.005, patience=10, verbose=0, mode='auto')

model.fit_generator(train_batches,steps_per_epoch = 100,validation_steps = 80,
                   validation_data=valid_batches, epochs=1000, verbose=1, callbacks=[csv_logger, early_stopping])

model.save(save_model_name)
