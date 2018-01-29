import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
import argparse, json

def parse():
    parser = argparse.ArgumentParser(description="Plant classifier")
    parser.add_argument('--uid', type=str, help='training uid', required=True)
    parser.add_argument('--train_path',type=str,  default='data/train', help='training data path')
    parser.add_argument('--valid_path',type=str, default='data/valid', help='valid data path')
    parser.add_argument('--train_resnet', default = True, action='store_true', help='whether train on ResNet50')
    parser.add_argument('--train_inception', action='store_true', help='whether train on InceptionResNetV2')
    parser.add_argument('--learning_rate', type=float, default=0.00008, help='learning rate for training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--img_size', type=int, default=224, help='img width, height size')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args



class PlantClassifier(object):
    ''' Initialize the parameters for the model '''
    def __init__(self, args):
        self.uid = args.uid
        self.train_path = args.train_path
        self.valid_path = args.valid_path
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.weights = 'imagenet'
        self.trainable = False
        self.lr = args.learning_rate
        self.epochs = 1
        self.num_classes = 240
        self.model = self.create_base_model(args)
        self.model_name = ""
        self.save_model_path = ""
        self.save_log_path = ""


        if(args.train_resnet):
            self.model_name = 'ResNet50_' + self.uid
        elif(args.train_inception):
            self.model_name = 'InceptionResNetV2_' + self.uid

        self.init_files(args)

    def init_files(self, args):
        if not (os.path.exists("model/%s/" % self.model_name)):
            os.makedirs("model/%s/" % self.model_name)
        with open(os.path.join('model', self.model_name, 'paras.json'), "w") as f:
            json.dump(vars(args), f)

    def ResNet50_model(self, w = 'imagenet', trainable = False):
        model = ResNet50(weights=w, include_top=False, input_shape=(self.img_size , self.img_size , 3))
        if(not trainable):
            for layer in model.layers:
                layer.trainable = False
        return model

    def InceptionResNetV2_model(self, w = 'imagenet', trainable = False):
        model = InceptionResNetV2(weights=w, include_top=False, input_shape=(self.img_size , self.img_size , 3))
        if(not trainable):
            for layer in model.layers:
                layer.trainable = False
        return model

    def VGG16_model(self):
        vgg16_model = keras.applications.vgg16.VGG16()
        model = Sequential()
        for layer in vgg16_model.layers:
            model.add(layer)
        model.layers.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(4096, activation='relu', name='fc3'))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def rebase_base_model(self,model):
        for layer in model.layers:
            layer.trainable = True
        return model

    def add_custom_layers(self, base_model):
        x = base_model.output
        x = Flatten()(x)
        # x = Dense(2048, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        x = Dense(4096, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        x =  LeakyReLU(0.2)(x)
        # x = Dense(2048, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        y = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=y)
        return model
    
    def create_base_model(self, args):
        if args.train_inception:
            model = self.InceptionResNetV2_model(w = self.weights , trainable = self.trainable)
            model = self.add_custom_layers(model)
            return model
        elif args.train_resnet:
            model = self.ResNet50_model(w = self.weights, trainable = self.trainable)
            model = self.add_custom_layers(model)
            return model

    def train(self, trainable=False):
        if(not trainable):
            self.save_model_path = os.path.join('model' , self.model_name, self.model_name + '.h5')
            self.save_log_path =  os.path.join('model' , self.model_name, self.model_name + '_log.csv')
        else:
            self.model = load_model(self.save_model_path)
            self.save_model_path = os.path.join('model' , self.model_name, self.model_name + '_2.h5')
            self.save_log_path =  os.path.join('model' , self.model_name, self.model_name + '_log.csv')
        sv = ModelCheckpoint(self.save_model_path,
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                )
        train_batches = ImageDataGenerator(
            # rotation_range=1,
            # width_shift_range=0.1,
            # height_shift_range=0.1,
            # shear_range=0.05,
            # zoom_range=0.1,
            # fill_mode='nearest',
            # horizontal_flip=True,
            # vertical_flip=False
        ).flow_from_directory(self.train_path, target_size=(self.img_size ,self.img_size ), batch_size=self.batch_size)
        valid_batches = ImageDataGenerator().flow_from_directory(self.valid_path, target_size=(self.img_size ,self.img_size ), batch_size=self.batch_size)

        # parallel_model = multi_gpu_model(model, gpus=2)
        self.model.compile(Adam(lr=self.lr), loss='categorical_crossentropy', metrics=['accuracy'])

        csv_logger = CSVLogger(self.save_log_path, append=True, separator=',')
        early_stopping = EarlyStopping(monitor='val_acc', patience=10)
        self.model.fit_generator(train_batches,steps_per_epoch = 100,validation_steps = 80,
                        validation_data=valid_batches, epochs=self.epochs, verbose=1, callbacks=[csv_logger, early_stopping, sv])
        self.model.save(self.save_model_path)

    def evaluate(self):
        if(os.path.exists(self.save_model_path)):
            model = load_model(self.save_model_path)
        else:
            return
        valid_batches = ImageDataGenerator().flow_from_directory(self.valid_path, target_size=(self.img_size ,self.img_size ), batch_size=self.batch_size)
        scores = model.evaluate_generator(valid_batches, steps= len(valid_batches))
        print("loss, acc", scores)

args = parse()
pc = PlantClassifier(args)
pc.train(trainable=False)
pc.rebase_base_model(pc.model)
pc.batch_size = 16
pc.train(trainable=True)
pc.evaluate()