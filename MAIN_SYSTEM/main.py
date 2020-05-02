from run_training import run
import sys
import argparse

# Get function from evaluate_model
from evaluate_model import retrieve_test_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_name', help='Resnet50/Resnet18/VGG19/VGG16')
    parser.add_argument('train_file', help='Path to train file')
    parser.add_argument('val_file', help='Path to validation file')
    parser.add_argument('--test_file', default="", help='Path to test file')
    parser.add_argument('--dropout', '-d', type=float, default=0, help="Dropout Value")
    parser.add_argument('--epoch', '-e', type=int, default=100, help="Epoch of training")
    parser.add_argument('--class_number', '-c', type=int, default=5, help="Number of classes/labels")
    parser.add_argument('--batch_size', '-b', type=int, default=32, help="batch size")
    parser.add_argument('--optimizer', '-o', default='adam', help='Optimizer for model')
    parser.add_argument('--lr_value', '-lr', default=1e-3, type=float, help='learning rate value for optimizer')
    
    args = parser.parse_args()
    modelname = args.model_name
    trainfile = args.train_file
    valfile = args.val_file
    testfile = args.test_file
    dropout = args.dropout
    epoch = args.epoch
    numclasses = args.class_number
    batchsize = args.batch_size
    lrvalue = args.lr_value
    optimizer = args.optimizer

    curr_model, x_val, y_val = run(modelname.lower(), trainfile, valfile, int(numclasses), dropout,  int(epoch), int(batchsize), float(lrvalue), optimizer.lower())
    
    # evaluate on val data
    loss, acc = curr_model.evaluate(x_val, y_val)
    print(f"evaluate {modelname} using validation data:\n loss: {loss}\n acc: {acc}")
    
    # evaluate on test data
    if testfile != "":
        x_test, y_test = retrieve_test_dataset(testfile, numclasses)
        loss, acc = curr_model.evaluate(x_test, y_test)
        print(f"evaluate {modelname} using testing data:\n loss: {loss}\n acc: {acc}")
