import sys
import h5py
import numpy as np
from my_resnet50 import ResNet50
from dataset_processing_utils import convert_to_one_hot
from keras import optimizers

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

def main(path_weight_file, test_file, class_num):
    """
    main function to print model evaluation result

    Arguments:\n
    class_num --> Int, number of class/label\n
    path_weight_file --> String, path which store .hdf5 of model's weight\n
    test_file --> String, path to which store .h5 file of test dataset

    Returns:\n
    none
    """
    # Init new model as ResNet50
    new_model = ResNet50(input_shape=(224, 224, 3), classes=int(class_num))
    
    # Load model weights
    new_model.load_weights(path_weight_file)

    # retrieve X_test, Y_test
    X_test, Y_test = retrieve_test_dataset(test_file)

    # Evaluate the model
    adam = optimizers.Adam(lr=0.0001)
    new_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    loss, acc = new_model.evaluate(X_test, Y_test)
    print(f"loss: {loss},   acc: {acc}")

if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Error. please check your arguments:")
        print("python evaluate_model.py [path_weight_file] [test_file] [class_num] ")    