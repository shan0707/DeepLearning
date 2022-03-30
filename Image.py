import numpy as np
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,  # 图像列表
                                  labels,  # 标签列表
                                  index,  # 从index个开始显示
                                  num=10):  # 缺省一次显示10幅
    fig = plt.gcf()  # 获取当前图表
    fig.set_size_inches(10, 12)  # 显示成英寸（1英寸等于2.54cm）
    if num > 25:
        num = 25  # 最多显示25幅图片
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)  # 画多个子图（5*5）

        ax.imshow(np.reshape(images[index], (1, 10)), cmap='binary')  # 显示第index张图像

        title = "label=" + str(np.argmax(labels[index]))  # 构建图片上要显示的title
        #if len(prediction) > 0:
            #title += ", predict=" + str(prediction[index])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])  # 不显示坐标轴
        ax.set_yticks([])
        index += 1
    plt.show()

