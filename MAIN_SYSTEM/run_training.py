# load requirements for model resnet50 and resnet18
from my_resnet50 import *
from my_resnet18 import *
from dataset_processing_utils import *
import matplotlib.pyplot as plt

# load requirements for pretrained model (vgg16, vgg19)
from keras import applications
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D

def run(model_name, train_file, val_file, num_classes, dropout_value, epoch = 100, batch_size = 32, lr_value = 1e-3 , optimizer = "adam", input_shape_arg = (224,224,3)):
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
        optimizer --> 1 for Adam, 2 for SGD\n

        Returns:\n
        model\n
        x_test\n
        y_test
    """

    # configure optimizer
    if optimizer == "adam":
        myoptimizer = optimizers.Adam(lr=lr_value)
    elif optimizer == "sgd":
        myoptimizer = optimizers.SGD(lr=lr_value, momentum=0.9, decay=1e-6, nesterov=True)
    else:
        myoptimizer = optimizers.Adam(lr=lr_value)
    print(f"{optimizer}: {lr_value}")

    # compile model
    if model_name == "resnet50":
        print("resnet50")
        model = ResNet50(input_shape=input_shape_arg, classes=int(num_classes), dropout_value=dropout_value)
        model.compile(optimizer=myoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    elif model_name == "resnet18":
        print("resnet18")
        model = ResNet18(input_shape=input_shape_arg, classes=int(num_classes), dropout_value=dropout_value)
        model.compile(optimizer=myoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    elif model_name == "vgg19":
        print("VGG19")
        # configure model
        base_model = applications.vgg19.VGG19(weights= None, include_top=False, input_shape= input_shape_arg)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout_value)(x)
        out = Dense(int(num_classes), activation= 'softmax')(x)
        model = Model(inputs = base_model.input, outputs = out)
        model.compile(optimizer= myoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    elif model_name == "vgg16":
        print("VGG16")
        # configure model
        base_model = applications.vgg16.VGG16(weights= None, include_top=False, input_shape= input_shape_arg)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout_value)(x)
        out = Dense(int(num_classes), activation= 'softmax')(x)
        model = Model(inputs = base_model.input, outputs = out)
        model.compile(optimizer= myoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # preprocessing data
    X_train, Y_train, X_val, Y_val = dataset_preprocess(num_classes, train_file, val_file)

    # set checkpoint 
    ckpt_filename = f"my{model_name}-{batch_size}-{optimizer}-{lr_value}.hdf5"
    checkpoint = ModelCheckpoint(ckpt_filename,
                                monitor='val_accuracy',
                                save_best_only="True",
                                # save_weights_only="True",
                                verbose=1)

    # set early stopping
    earlyStopping = EarlyStopping(monitor='val_accuracy', 
                                  mode='max', 
                                  verbose=1, 
                                  patience=10)

    if optimizer == "sgd":
        # reduce lr for SGD Optimizer
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1e-1,
                                patience=5, min_lr=1e-7, verbose=1)

        # train the model
        history = model.fit(X_train, Y_train, epochs = epoch, batch_size = batch_size, 
                validation_data=(X_val, Y_val), 
                shuffle=True,
                callbacks=[checkpoint, earlyStopping, reduce_lr])
    else: # optimizer == adam
        # train the model
        history = model.fit(X_train, Y_train, epochs = epoch, batch_size = batch_size, 
                validation_data=(X_val, Y_val), 
                shuffle=True,
                callbacks=[checkpoint, earlyStopping])
    

    # create graph
    f, axarr = plt.subplots(nrows=1,ncols=2, figsize=(12,4))

    plt.sca(axarr[0])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'acc-{model_name}-{batch_size}-{optimizer}({lr_value})')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    for val in [0.8, 0.9]:
        plt.axhline(y=val,color='gray',linestyle='--')

    plt.sca(axarr[1])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'loss-{model_name}-{batch_size}-{optimizer}({lr_value})')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    for val in [1.0]:
        plt.axhline(y=val,color='gray',linestyle='--')

    plt.savefig(f"{model_name}-{batch_size}-{optimizer}-{lr_value}.png")

    # return model, and validation data
    return model, X_val, Y_val