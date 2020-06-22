# load requirements for model resnet50 and resnet18
from my_resnet50 import *
from my_resnet18 import *
from dataset_processing_utils import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from hyperparameter import Hyperparameter

import argparse
import statistics

# load requirements for pretrained model (vgg16, vgg19)
from keras import applications
from keras import optimizers
import keras.metrics as metrics
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras import backend as K

def run(model_name, train_file, val_file, num_classes, filename, dropout, input_shape_arg = (224,224,3)):
    """
        fit dataset and run training process

        Arguments:\n
        train_file -->  h5 file, fit to model\n
        val_file --> h5 file, for validation\n
        num_classes --> int, total classes \n
        dropout_value --> float, range 0 - 1 for dropout\n
        epoch --> int\n
        batch_size --> int, [8, 16, 32, 64, 128, 256, etc.]\n
        input_shape_arg --> shape of image (W,H,C)\n
        lr_value --> learning rate value\n
        optimizer --> Adam, SGD\n

        Returns:\n
        model\n
        x_test\n
        y_test
    """
    
    # preprocessing data
    X_train, Y_train, X_val, Y_val = dataset_preprocess(num_classes, train_file, val_file)

    _epoch = 80
    lr_value_array = [1e-3, 1e-4]
    if model_name == "resnet50":
        batch_size_array = [8, 16, 32]
        LABEL = ["e-3(8)", "e-3(16)", "e-3(32)", "e-4(8)", "e-4(16)", "e-4(32)"]
    elif model_name == "resnet18":
        batch_size_array = [16, 32, 64]
        LABEL = ["e-3(16)", "e-3(32)", "e-3(64)", "e-4(16)", "e-4(32)", "e-4(64)"]

    HP = []
    for lr in lr_value_array:
        for bs in batch_size_array:
            hp = Hyperparameter("adam", lr, bs, dropout)
            HP.append(hp)

    HISTORY = []
    ROC = []
    for hp in HP:
        K.clear_session()
        model = None
        # compile model
        if model_name == "resnet50":
            print("resnet50")
            model = ResNet50(input_shape=input_shape_arg, classes=int(num_classes), dropout_value=hp.dropout)
            model.compile(optimizer=hp.get_optimizer(), loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC(), metrics.Precision(), metrics.Recall()])
        elif model_name == "resnet18":
            print("resnet18")
            model = ResNet18(input_shape=input_shape_arg, classes=int(num_classes), dropout_value=hp.dropout)
            model.compile(optimizer=hp.get_optimizer(), loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC(), metrics.Precision(), metrics.Recall()])
        elif model_name == "vgg19":
            print("VGG19")
            # configure model input    
            base_model = applications.vgg19.VGG19(weights= None, include_top=False, input_shape= input_shape_arg)
            # configure model output
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(hp.dropout(x))
            out = Dense(int(num_classes), activation= 'softmax')(x)
            # combine model then compile
            model = Model(inputs = base_model.input, outputs = out)
            model.compile(optimizer= hp.get_optimizer(), loss='categorical_crossentropy', metrics=['accuracy'])
        elif model_name == "vgg16":
            print("VGG16")
            # configure model input
            base_model = applications.vgg16.VGG16(weights= None, include_top=False, input_shape= input_shape_arg)
            # configure model output
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(hp.dropout(x))
            out = Dense(int(num_classes), activation= 'softmax')(x)
            # combine model then compile
            model = Model(inputs = base_model.input, outputs = out)
            model.compile(optimizer= hp.get_optimizer(), loss='categorical_crossentropy', metrics=['accuracy'])

        # optimizer == adam
        # train the model
        history = model.fit(X_train, Y_train, epochs = _epoch, batch_size = hp.batch_size, 
                validation_data=(X_val, Y_val), 
                shuffle=True)
        HISTORY.append(history)
        del model

        print(f"DONE for: {hp.optim}-{hp.lr_value}-{hp.batch_size}-{hp.dropout}")

    
    plt.figure(1)
    mpl.style.use('seaborn')
    i = 0
    for history in HISTORY:
        plt.plot(history.history["val_accuracy"], f"C{i}", label=LABEL[i])
        i = i+1
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.title(f"Accuracy {filename}")
    plt.legend()
    plt.savefig(f"ACC-{filename}.png")

    print("VAL ACC:")
    i = 0
    for history in HISTORY:
        print(LABEL[i])
        i = i + 1
        print("auc: {}" .format(statistics.mean( history.history["val_auc_1"] )) )
        print("recall: {}" .format(statistics.mean( history.history["val_recall_1"] )) )
        print("Prec: {}" .format(statistics.mean( history.history["val_precision_1"] )) )
        print("MEAN: {}" .format(statistics.mean( history.history["val_accuracy"] )) )
        print("STD: {}" .format(statistics.pstdev( history.history['val_accuracy'] )) )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start Tuning', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_name', help='Resnet50/Resnet18/VGG19/VGG16')
    parser.add_argument('train_file', help='Path to train file')
    parser.add_argument('val_file', help='Path to validation file')
    parser.add_argument('--dropout', '-d', type=float, help='dropout_value')
    parser.add_argument('--filename', '-f', default="default", help='file name for png')
    parser.add_argument('--class_number', '-c', type=int, default=5, help="Number of classes/labels")
    
    args = parser.parse_args()
    modelname = args.model_name
    trainfile = args.train_file
    filename = args.filename
    valfile = args.val_file
    numclasses = args.class_number
    dropout = args.dropout

    run(modelname.lower(), trainfile, valfile, int(numclasses), filename, dropout)