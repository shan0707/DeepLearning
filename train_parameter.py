import numpy as np
from TwoLayerNet import TwoLayerNet
from mnist import load_mnist
#第二步，训练超参数
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_val = x_train[50000:60000, :]
t_val = t_train[50000:60000, :]
x_train = x_train[0:50000, :]
t_train = t_train[0:50000, :]
best_net = None  # store the best model into this
results = {}
best_val_acc = -1
# some params are fixed, others are varied
batch_size = 100
lr_range = np.linspace(0.1, 0.4, 2)
reg_range = np.linspace(1e-5, 1e-4, 2)
hidden_size_range = np.linspace(50, 60, 5)
best_params = None
for lr in lr_range:
    for reg in reg_range:
        for hidden_size in hidden_size_range:
            net = TwoLayerNet(int(hidden_size), input_size=784, output_size=10, weight_init_std=0.01)
            stats = net.train(x_train, t_train, x_val, t_val, lr, reg, lr_decay=0.95, batch_size=100)
            train_acc = stats['train_acc']
            val_acc = stats['val_acc']
            print('Validation accuracy: ', val_acc)
            results[(lr, reg, hidden_size)] = val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = net
                best_params = (lr, reg, hidden_size)
print(best_params)
