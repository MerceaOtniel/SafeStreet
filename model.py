from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalAveragePooling1D
from keras.utils import plot_model, np_utils
from keras import optimizers
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import argparse
from PIL import Image
from keras.layers import Dense, Input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.optimizers import Nadam
from keras.applications.vgg16 import VGG16
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint

batch_size = 16
num_classes = 2
epochs = 10
frames = 3

def build_model():    
    video = Input(shape=(frames,224,224,3))
    cnn_base = VGG16(input_shape=(224,224,3),weights="imagenet",include_top=False)
    cnn_out = GlobalAveragePooling2D()(cnn_base.output)
    cnn = Model(input=cnn_base.input, output=cnn_out)
    cnn.trainable = False
    encoded_frames = TimeDistributed(cnn)(video)
    encoded_sequence = LSTM(500)(encoded_frames)
    hidden_layer = Dense(output_dim=2048, activation="relu")(encoded_sequence)
    outputs = Dense(output_dim=num_classes, activation="softmax")(hidden_layer)
    model = Model([video], outputs)
    optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"]) 
    return model


def batch_iter(split_file):
    split_data = np.genfromtxt(split_file, dtype=None, delimiter=" ")
    total_seq_num = len(split_data)
    num_batches_per_epoch = int((total_seq_num - 1) / batch_size) + 1

    def data_generator():
        while 1:
            indices = np.random.permutation(np.arange(total_seq_num))

            for batch_num in range(num_batches_per_epoch): 
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, total_seq_num)

                RGB = []
                Y = []
                
                for i in range(start_index, end_index): 
                    image_dir = split_data[indices[i]][0].decode("UTF-8")
                    seq_len = int(split_data[indices[i]][1])
                    y = int(split_data[indices[i]][2])
                    augs_rgb = []
                    
                    for j in range(frames): 
                        frame = int(seq_len / frames * j) + 1
                        rgb_i = load_img("%s/out%d.png" % (image_dir, frame), target_size=(224, 224))
                        rgb = img_to_array(rgb_i)
                        rgb_flip_i = rgb_i.transpose(Image.FLIP_LEFT_RIGHT) # augmentation
                        rgb_flip = img_to_array(rgb_flip_i)
                        augs_rgb.append([rgb, rgb_flip])
                        
                    augs_rgb = np.array(augs_rgb).transpose((1, 0, 2, 3, 4))
                    RGB.extend(augs_rgb)
                    Y.extend([y, y])
                RGB = np.array(RGB)
                RGB = RGB.astype('float32') / 255
                Y = np_utils.to_categorical(Y, num_classes)
                yield ([RGB], Y)

    return num_batches_per_epoch, data_generator()


def plot_history(history):
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    


if __name__ == "__main__":
   
   
    train_split_file = "drive/hacktm2018sibiu/data/data_train_shuffle.txt"
    test_split_file = "drive/hacktm2018sibiu/data/data_validation_shuffle.txt" 

    if not os.path.exists("model"):
        os.makedirs("model")

    model = build_model()
    model.summary()
    print("Built model")

    # Make batches
    train_steps, train_batches = batch_iter(train_split_file)
    valid_steps, valid_batches = batch_iter(test_split_file)
    
    checkpoint = ModelCheckpoint("drive/hacktm2018sibiu/model/cnn_lstm.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacksList=[checkpoint]

    # Train model
    history = model.fit_generator(train_batches, steps_per_epoch=train_steps*2,
                epochs=epochs, verbose=1, validation_data=valid_batches,
                callbacks=callbacksList,
                validation_steps=valid_steps*2)
    plot_history(history)
    print("Trained model")


    # Evaluate model
    score = model.evaluate_generator(valid_batches, valid_steps)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Clear session
    from keras.backend import tensorflow_backend as backend
    backend.clear_session()