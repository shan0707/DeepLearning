import numpy as np
from functions import sigmoid, sigmoid_grad, softmax, cross_entropy_error




class TwoLayerNet:

    def __init__(self, hidden_size, input_size, output_size, weight_init_std):
        # 初始化权重
        self.dict = {}
        self.dict['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 0.01*(784,50)
        self.dict['b1'] = np.zeros(hidden_size)  # (0......0) 1*50
        self.dict['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)  # 0.01*(50,10)
        self.dict['b2'] = np.zeros(output_size)  # (0......0) 1*10

    def predict(self, x):
        w1, w2 = self.dict['w1'], self.dict['w2']
        b1, b2 = self.dict['b1'], self.dict['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        return y

    def loss(self, y, t, reg=5e-6):
        w1, w2 = self.dict['w1'], self.dict['w2']
        t = t.argmax(axis=1)#数组t中每一行最大值所在“列”索引值
        num = y.shape[0]#样本数
        s = y[np.arange(num), t]#一共有num行，每一行对应的最大索引放在t内，y（i,j）表示第i个样本最大的
        loss = -np.sum(np.log(s)) / num
        loss += reg * (np.sum(w1 * w1) + np.sum(w2 * w2))
        return loss

    def gradient(self, x, t, reg=5e-6):
        w1, w2 = self.dict['w1'], self.dict['w2']
        b1, b2 = self.dict['b1'], self.dict['b2']
        grads = {}

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        num = x.shape[0]
        dy = (y - t) / num
        grads['w2'] = np.dot(z1.T, dy) + reg * w2
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, w2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['w1'] = np.dot(x.T, da1) + reg * w1
        grads['b1'] = np.sum(da1, axis=0)

        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        p = np.argmax(y, axis=1)
        q = np.argmax(t, axis=1)
        acc = np.sum(p == q) / len(y)  # t为精确值
        return acc

    def train(self, x_train, t_train, x, t, lr=0.1, reg=5e-6, lr_decay=0.95, batch_size=100, ):
        epoch = 20000
        train_size = x_train.shape[0]  # 50000
        iter_per_epoch = max(train_size / batch_size, 1)  # 500

        train_loss_list = []
        x_loss_list = []
        train_acc_list = []
        x_acc_list = []

        for i in range(epoch):
            batch_mask = np.random.choice(train_size, batch_size)  # 从0到50000 随机选100个数
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            grad = self.gradient(x_batch, t_batch, reg)

            for key in ('w1', 'b1', 'w2', 'b2'):
                self.dict[key] -= lr * grad[key]


            # 每批数据记录一次精度和当前的损失值
            if i % iter_per_epoch == 0:
                train_acc = self.accuracy(x_train, t_train)
                x_acc = self.accuracy(x, t)
                train_acc_list.append(train_acc)
                x_acc_list.append(x_acc)

                train_loss = self.loss(self.predict(x_train), t_train, reg)
                train_loss_list.append(train_loss)
                val_loss = self.loss(self.predict(x), t, reg)
                x_loss_list.append(val_loss)

                lr *= lr_decay
               # print(
               #     '第' + str(i + 1) + '次迭代''train_acc, test_acc, loss :| ' + str(train_acc) + ", " + str(
               #         val_acc) + ',' + str(
               #        train_loss))
        return {
                    'train_acc': train_acc,
                    'x_acc': x_acc,
                    'train_acc_list': train_acc_list,
                    'x_acc_list': x_acc_list,
                    'train_loss_list': train_loss_list,
                    'x_loss_list': x_loss_list,
        }