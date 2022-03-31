import numpy as np
import matplotlib.pyplot as plt
from TwoLayerNet import TwoLayerNet
from functions import sigmoid
from mnist import load_mnist
from Image import plot_images_labels_prediction
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


x_val = x_train[50000:60000, :]
t_val = t_train[50000:60000, :]
x_train = x_train[0:50000, :]
t_train = t_train[0:50000, :]
#第一步：利用训练集，验证集训练网络
#net = TwoLayerNet(hidden_size=64, input_size=784, output_size=10, weight_init_std=0.01)
#stats = net.train(x_train, t_train, x_val, t_val, lr=0.1, reg=1e-06, lr_decay=0.95, batch_size=100)

#第三步；利用测试集测试最优网络
net = TwoLayerNet(hidden_size=64, input_size=784, output_size=10, weight_init_std=0.01)
stats = net.train(x_train, t_train, x_test, t_test, lr=0.4, reg=1e-05, lr_decay=0.95, batch_size=100)

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(stats['train_acc_list']))
plt.plot(x, stats['train_acc_list'], label='train acc')
plt.plot(x, stats['x_acc_list'], label='test acc', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(stats['train_loss_list']))
plt.plot(x, stats['train_loss_list'], label='train loss')
plt.plot(x, stats['x_loss_list'], label='test loss', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title('Loss')
plt.show()

'''
#第四步：可视化网络超参数，从第1张照片开始显示，显示10张（注意修改predict的返回值）
w1, w2 = net.dict['w1'], net.dict['w2']
b1, b2 = net.dict['b1'], net.dict['b2']
a1 = np.dot(x_test, w1) + b1
z1 = sigmoid(a1)
a2 = np.dot(z1, w2) + b2
plot_images_labels_prediction(a2, t_test, 1, 10)
'''

