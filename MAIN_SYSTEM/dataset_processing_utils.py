import h5py
import numpy as np

def load_dataset(train_file, test_file=""):
    """
    Load train and test dataset from h5 file

    Arguments:\n
    train_file --> String, full path location of train file in .h5 format\n
    test_file --> String, full path location of test file in .h5 format\n

    Returns:\n
    train_set_x_orig --> (train dataset) array of images in shape: (N, Height, Weight, Channel)\n
    train_set_y_orig --> (train dataset) array of labels in shape: (N, label)\n
    test_set_x_orig --> (train dataset) array of images in shape: (N, Height, Weight, Channel)\n
    test_set_y_orig --> (train dataset) array of labels in shape: (N, label)\n
    """
    train_dataset = h5py.File(train_file, "r")
    train_set_x_orig = np.array(train_dataset["x_images"][:]) 
    train_set_y_orig = np.array(train_dataset["y_labels"][:]) 
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    if test_file != "":
        test_dataset = h5py.File(test_file, "r")
        test_set_x_orig = np.array(test_dataset["x_images"][:]) 
        test_set_y_orig = np.array(test_dataset["y_labels"][:]) 
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig 
    
    return train_set_x_orig, train_set_y_orig

def convert_to_one_hot(Y, C):
    """
    convert array to one hot matrix

    Arguments:
    Y --> Array, array of label image, shape: (N, )
    C --> Integer, number of classes/labels for classification

    Return:
    Y --> one hot matrix contain labels for each row
    """
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def dataset_preprocess(num_class, train_file, test_file=""):
    """
    preprocess dataset before fit to model\n

    Arguments:\n
    num_class -->   Int, number of labels\n
    train_file -->  H5 file, contain train dataset (X, y)\n
    test_file -->   H5 file, contain test dataset (X, y)\n

    Returns:\n
    X_train --> Array (N, height, width, channels), train images which ready to fit to model\n
    Y_train --> Array (N, ), train lables which ready to fit to model\n
    X_test --> Array (N, height, width, channels)\n
    Y_test --> Array (N, )\n
    """

    # loading dataset
    if test_file != "":
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset(train_file, test_file) 
    else:
        X_train_orig, Y_train_orig = load_dataset(train_file, test_file) 

    # Normalize image vectors
    X_train = X_train_orig/255.
    if test_file != "": X_test = X_test_orig/255. 

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, num_class).T
    if test_file != "": Y_test = convert_to_one_hot(Y_test_orig, num_class).T 

    print ("number of training examples = " + str(X_train.shape[0]))
    if test_file != "": print ("number of test examples = " + str(X_test.shape[0])) 
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    if test_file != "": print ("Y_test shape: " + str(Y_test.shape)) 
    if test_file != "": print ("X_test shape: " + str(X_test.shape)) 

    if test_file != "": return X_train, Y_train, X_test, Y_test
    return X_train, Y_train 