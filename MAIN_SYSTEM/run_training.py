from my_resnet50 import *
from my_resnet18 import *
from dataset_processing_utils import *
import matplotlib.pyplot as plt

def run(resnetlayer, train_file, test_file, num_classes, epoch = 100, batch_size = 32, lr_value = 1e-3 , optimizer = 1, input_shape_arg = (224,224,3)):
    """
        fit dataset and run training process

        Arguments:\n
        train_file -->  h5 file\n
        test_file --> h5 file\n
        num_classes --> int, total classes \n
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
    if optimizer == 1:
        myoptimizer = optimizers.Adam(lr=lr_value)
    elif optimizer == 2:
        myoptimizer = optimizers.SGD(lr=lr_value, momentum=0.9, decay=1e-4)
    else:
        myoptimizer = optimizers.Adam(lr=lr_value)

    # compile model
    if resnetlayer == "resnet50":
        print("resnet50")
        model = ResNet50(input_shape=input_shape_arg, classes=int(num_classes))
        # for layer in [l for l in model.layers if l.name in ['res3d', 'res4'] or 'res5' in l.name]:
        #     layer.trainable = False
        model.compile(optimizer=myoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    elif resnetlayer == "resnet18":
        print("resnet18")
        model = ResNet18(input_shape=input_shape_arg, classes=int(num_classes))
        model.compile(optimizer=myoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # preprocessing data
    X_train, Y_train, X_test, Y_test = dataset_preprocess(num_classes, train_file, test_file)

    # set checkpoint 
    ckpt_filename = f"my{resnetlayer}-{batch_size}-{optimizer}-{lr_value}.hdf5"
    checkpoint = ModelCheckpoint(ckpt_filename,
                                monitor='val_accuracy',
                                save_best_only="True",
                                save_weights_only="True",
                                verbose=1)

    # set early stopping
    earlyStopping = EarlyStopping(monitor='val_accuracy', 
                                  mode='max', 
                                  verbose=1, 
                                  patience=12)

    # train the model
    history = model.fit(X_train, Y_train, epochs = epoch, batch_size = batch_size, 
              validation_data=(X_test, Y_test), 
              shuffle=True,
              callbacks=[checkpoint, earlyStopping])

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

    plt.savefig(f"my{resnetlayer}-{batch_size}-{optimizer}-{lr_value}.png")

    # return model, and testing data
    return model, X_test, Y_test