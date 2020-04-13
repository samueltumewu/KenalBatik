# all stuff for developing deep learning model 
import sys
import h5py
import numpy as np
from dataset_processing_utils import convert_to_one_hot
import matplotlib.pyplot as plt
import os  
from keras import optimizers
from keras import applications
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint

def retrieve_test_dataset(test_file):
    """
    load and preprocess test dataset from .h5 file

    Arguments:\n
    test_file --> String, path to which store h5 file of test dataset

    Returns:\n
    X_test --> set X of test dataset
    Y_test --> set Y of test dataset
    """
    # loading set x and set y of dataset
    test_dataset = h5py.File(test_file, "r")
    test_set_x_orig = np.array(test_dataset["x_images"][:]) 
    test_set_y_orig = np.array(test_dataset["y_labels"][:]) 

    # reshape set 'y' 
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # Normalize image vectors
    X_test = test_set_x_orig/255.
    Y_test = test_set_y_orig/255.
    
    # Convert test set Y to one hot matrix
    Y_test = convert_to_one_hot(test_set_y_orig, 5).T
    
    return X_test, Y_test 

def main(model_code, path_weight_file, test_file, class_num):
    """
    main function to print model evaluation result

    Arguments:\n
    model_code --> Int, 1 for resnet50, 2 for vgg16, 3 for vgg19
    class_num --> Int, number of class/label\n
    path_weight_file --> String, path which store .hdf5 of model's weight\n
    test_file --> String, path to which store .h5 file of test dataset

    Returns:\n
    none
    """
    if int(model_code) == 1:
        print("model code 1")
    elif int(model_code) == 2:
        print("loading vgg16...")
        img_height,img_width = 224,224 
        num_classes = int(class_num)
        base_model = applications.vgg16.VGG16(weights= None, include_top=False, input_shape= (img_height,img_width,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        out = Dense(num_classes, activation= 'softmax')(x)
        new_model = Model(inputs = base_model.input, outputs = out)
        evaluate(new_model, path_weight_file, test_file)
    elif int(model_code) == 3:
        print("loading vgg19...")
        img_height,img_width = 224,224 
        num_classes = int(class_num)
        base_model = applications.vgg19.VGG19(weights= None, include_top=False, input_shape= (img_height,img_width,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        out = Dense(num_classes, activation= 'softmax')(x)
        new_model = Model(inputs = base_model.input, outputs = out)
        evaluate(new_model, path_weight_file, test_file)

def evaluate(model, path_weight_file, test_file):
    # assign model
    new_model = model

    # Load model weights
    new_model.load_weights(path_weight_file)

    # retrieve X_test, Y_test
    X_test, Y_test = retrieve_test_dataset(test_file)

    # Evaluate the model
    adam = optimizers.Adam(lr=1e-4)
    new_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    loss, acc = new_model.evaluate(X_test, Y_test)
    print(f"loss: {loss},   acc: {acc}")

if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Error. please check your arguments:")
        print("python evaluate_pretrained_model.py [model code] [path_weight_file] [test_file] [class_num] ")    