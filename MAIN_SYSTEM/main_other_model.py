from run_pretrained import *
import sys

def main(model_code, train, test, num_class, batch_size=32, optimizer_code=1, lr_value=1e-4):
    """
    Features:
        1. Execute fit function to training dataset
        2. Evaluate model
        3. print model loss and acc 
    """

    if int(model_code) == 1:
        print(f"batch_size: {batch_size}\noptimizer_code: {optimizer_code}\nlr_value: {lr_value}")
        my_model, x_test, y_test = run_resnet50(train, test, int(num_class), 100, int(batch_size), int(optimizer_code), float(lr_value))
        loss, acc = my_model.evaluate(x_test, y_test)
        print(f"evaluate model using validation data:\n loss: {loss}\n acc: {acc}")
    elif int(model_code) == 2:
        print(f"batch_size: {batch_size}\noptimizer_code: {optimizer_code}\nlr_value: {lr_value}")
        my_model, x_test, y_test = run_vgg16(train, test, int(num_class), 100, int(batch_size), int(optimizer_code), float(lr_value))
        loss, acc = my_model.evaluate(x_test, y_test)
        print(f"evaluate model using validation data:\n loss: {loss}\n acc: {acc}")
    elif int(model_code) == 3:
        print(f"batch_size: {batch_size}\noptimizer_code: {optimizer_code}\nlr_value: {lr_value}")
        my_model, x_test, y_test = run_vgg19(train, test, int(num_class), 100, int(batch_size), int(optimizer_code), float(lr_value))
        loss, acc = my_model.evaluate(x_test, y_test)
        print(f"evaluate model using validation data:\n loss: {loss}\n acc: {acc}")

if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 8:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    else:
        print("Error. please check your arguments:")
        print("python main_other_model.py [model code] [train file] [test file] [class number]")
        print("python main_other_model.py [model code] [train file] [test file] [class number] [batch_size] [optimizer_code] [lr_value]")