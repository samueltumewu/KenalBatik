from run_training import run
import sys

def main(resnetlayer, train, test, num_class=5, batch_size=32, lr_value=1e-4, optimizer=1):
    """
    Features:
        1. Execute fit function to training dataset
        2. Evaluate model
        3. print model loss and acc 
    """
    my_model, x_test, y_test = run(resnetlayer.lower(), train, test, int(num_class), 100, int(batch_size), float(lr_value), int(optimizer))
    loss, acc = my_model.evaluate(x_test, y_test)
    print(f"evaluate model using validation data:\n loss: {loss}\n acc: {acc}")

if __name__ == '__main__':
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 8:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    else:
        print("Error. please check your arguments:")
        print("python main.py [resnet layer] [train file] [test file] [class number]")
        print("python main.py [resnet layer] [train file] [test file] [class number] [batch_size] [lr_value] [optimizer]")