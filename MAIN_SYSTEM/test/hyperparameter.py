from keras import optimizers

class Hyperparameter:
    def __init__(self, optim="adam", lr_value=1e-3, batch_size=32, dropout=0.3):
        self.optim = optim
        self.lr_value = lr_value
        self.batch_size = batch_size
        self.dropout = dropout
    
    def get_optimizer(self):
        # configure optimizer
        if self.optim == "adam":
            myoptimizer = optimizers.Adam(lr=self.lr_value)
        elif self.optim == "sgd":
            myoptimizer = optimizers.SGD(lr=self.lr_value, momentum=0.9, decay=1e-6, nesterov=True)
        else:
            myoptimizer = optimizers.Adam(lr=self.lr_value)
        return myoptimizer

