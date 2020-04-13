"""
    this py file is to defined functions to run pretrained model from keras
    they are:
    1. Resnet 50
    2. Resnet 101
    3. VGG16
    4. VGG19
"""
# all stuff for developing deep learning model 
import numpy as np 
import matplotlib.pyplot as plt
import os  
from keras import applications
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint,EarlyStopping

# custom
from dataset_processing_utils import *
import matplotlib.pyplot as plt

def run_resnet50(train_file, test_file, num_classes, epoch = 100, batch_size = 32, input_shape_arg = (224,224,3), lr_value = 1e-4 , optimizer = 1):
    print("resnet50")
    # preprocessing data
    X_train, Y_train, X_test, Y_test = dataset_preprocess(num_classes, train_file, test_file)

    # configure optimizer
    if optimizer == 1:
        myoptimizer = optimizers.Adam(lr=lr_value)
    elif optimizer == 2:
        myoptimizer = optimizers.SGD(lr=lr_value)
    else:
        myoptimizer = optimizers.Adam(lr=lr_value)

    # configure model
    img_height,img_width = 224,224 
    num_classes = num_classes
    base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = out)
    model.compile(optimizer= myoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # set checkpoint 
    checkpoint = ModelCheckpoint(f"resnet50-{batch_size}-{optimizer}-{lr_value}.hdf5",
                                monitor='val_acc',
                                save_best_only="True",
                                save_weights_only="True",
                                verbose=1)

    # train the model
    history = model.fit(X_train, Y_train, epochs = epoch, batch_size = batch_size, 
              validation_data=(X_test, Y_test), 
              callbacks=[checkpoint])

    # create graph
    f, axarr = plt.subplots(nrows=1,ncols=2, figsize=(12,4))

    plt.sca(axarr[0])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'acc-{batch_size}-{optimizer}({lr_value})')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.sca(axarr[1])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'acc-{batch_size}-{optimizer}({lr_value})')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig(f"resnet50pretrained-{batch_size}-{optimizer}-{lr_value}.png")

    # return model, and testing data
    return model, X_test, Y_test

def run_vgg19(train_file, test_file, num_classes, epoch = 100, batch_size = 32, input_shape_arg = (224,224,3), lr_value = 1e-4 , optimizer = 1):
    print("vgg19")
    # preprocessing data
    X_train, Y_train, X_test, Y_test = dataset_preprocess(num_classes, train_file, test_file)

    # configure optimizer
    if optimizer == 1:
        myoptimizer = optimizers.Adam(lr=lr_value)
    elif optimizer == 2:
        myoptimizer = optimizers.SGD(lr=lr_value)
    else:
        myoptimizer = optimizers.Adam(lr=lr_value)

    # configure model
    img_height,img_width = 224,224 
    num_classes = num_classes
    base_model = applications.vgg19.VGG19(weights= None, include_top=False, input_shape= (img_height,img_width,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = out)
    model.compile(optimizer= myoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # set checkpoint 
    checkpoint = ModelCheckpoint(f"vgg19-{batch_size}-{optimizer}-{lr_value}.hdf5",
                                monitor='val_accuracy',
                                save_best_only="True",
                                save_weights_only="True",
                                verbose=1)

    # train the model
    history = model.fit(X_train, Y_train, epochs = epoch, batch_size = batch_size, 
              validation_data=(X_test, Y_test), 
              callbacks=[checkpoint])

    # create graph
    f, axarr = plt.subplots(nrows=1,ncols=2, figsize=(12,4))

    plt.sca(axarr[0])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'acc-{batch_size}-{optimizer}({lr_value})')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.sca(axarr[1])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'loss-{batch_size}-{optimizer}({lr_value})')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig(f"vgg19-{batch_size}-{optimizer}-{lr_value}.png")

    # return model, and testing data
    return model, X_test, Y_test

def run_vgg16(train_file, test_file, num_classes, epoch = 100, batch_size = 32, input_shape_arg = (224,224,3), lr_value = 1e-4 , optimizer = 1):
    print("vgg16")
    # preprocessing data
    X_train, Y_train, X_test, Y_test = dataset_preprocess(num_classes, train_file, test_file)

    # configure optimizer
    if optimizer == 1:
        myoptimizer = optimizers.Adam(lr=lr_value)
    elif optimizer == 2:
        myoptimizer = optimizers.SGD(lr=lr_value)
    else:
        myoptimizer = optimizers.Adam(lr=lr_value)

    # configure model
    img_height,img_width = 224,224 
    num_classes = num_classes
    base_model = applications.vgg16.VGG16(weights= None, include_top=False, input_shape= (img_height,img_width,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = out)
    model.compile(optimizer= myoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # set checkpoint 
    checkpoint = ModelCheckpoint(f"vgg16-{batch_size}-{optimizer}-{lr_value}.hdf5",
                                monitor='val_accuracy',
                                save_best_only="True",
                                save_weights_only="True",
                                verbose=1)

    # train the model
    history = model.fit(X_train, Y_train, epochs = epoch, batch_size = batch_size, 
              validation_data=(X_test, Y_test), 
              callbacks=[checkpoint])

    # create graph
    f, axarr = plt.subplots(nrows=1,ncols=2, figsize=(12,4))

    plt.sca(axarr[0])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'acc-{batch_size}-{optimizer}({lr_value})')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.sca(axarr[1])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'loss-{batch_size}-{optimizer}({lr_value})')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig(f"vgg16-{batch_size}-{optimizer}-{lr_value}.png")

    # return model, and testing data
    return model, X_test, Y_test