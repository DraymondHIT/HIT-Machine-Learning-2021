import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import permutations

# 超参数
mean = [[0, 8], [10, 4], [10, 20]]
cov = [np.diag([2, 2]), np.diag([2, 2]), np.diag([2, 2])]
train_size = 200


# 随机生成数据
def generate_data(k, means, covs, sample_num):
    """
    生成k类二维高斯分布的数据
    :param k: 类别数
    :param means: list[list]，代表每一类的均值
    :param sample_num: 每一类的样本数
    :return: 返回生成的数据
    """
    samples = np.zeros((1, 1))
    for i in range(k):
        data_temp = np.random.multivariate_normal(means[i], covs[i], sample_num)
        if i == 0:
            samples = data_temp
        else:
            samples = np.concatenate((samples, data_temp), axis=0)
    return samples


def initialization(X, k):
    dimension = X.shape[1]
    sample_size = X.shape[0]

    center = np.zeros((k, dimension))
    center[0, :] = X[0, :]
    _index = [0]
    i = 1
    while i < k:
        max_distance = 0
        max_index = 0
        for j in range(sample_size):
            if j in _index:
                continue
            temp_distance = 0
            for l in range(len(_index)):
                temp_distance += np.linalg.norm(X[j, :] - X[_index[l], :])
            if temp_distance > max_distance:
                max_distance = temp_distance
                max_index = j
        center[i, :] = X[max_index, :]
        _index.append(max_index)
        i += 1

    return center


def kmeans(X, k, epsilon=1e-5):
    """
    K-means算法实现
    """
    dimension = X.shape[1]
    sample_size = X.shape[0]

    # 初始化中心点坐标和标签
    center = np.zeros((k, dimension))
    label = np.zeros(sample_size)

    # 随机分类中心坐标
    # for i in range(k):
    #     center[i, :] = X[np.random.randint(0, sample_size), :]

    # 均值坐标初始化优化
    center = initialization(X, k)

    while True:
        distance = np.zeros(k)

        # 根据中⼼重新给每个点贴分类标签
        for i in range(sample_size):
            for j in range(k):
                distance[j] = np.linalg.norm(X[i, :] - center[j, :])
            label[i] = np.argmin(distance)  # 把距某点最近的中心点作为它的分类标签

        # 根据每个点的标签计算新的中⼼点坐标
        new_center = np.zeros((k, dimension))
        count = np.zeros((k, 1))

        for i in range(X.shape[0]):
            new_center[int(label[i]), :] += X[i, :]  # 对每个类的所有点坐标求和
            count[int(label[i]), 0] += 1

        # 计算新的中⼼点坐标
        new_center /= count

        if np.linalg.norm(new_center - center) < epsilon:
            break
        else:
            center = new_center

    return X, label, center


def showResult(X, label, center):
    plt.scatter(X[:, 0], X[:, 1], c=label, s=10)
    plt.scatter(center[:, 0], center[:, 1], marker='+', s=500)
    plt.show()


def accuracy(real_label, class_label, k):
    classes = list(permutations(range(k), k))
    counts = np.zeros(len(classes))
    for i in range(len(classes)):
        for j in range(real_label.shape[0]):
            if int(real_label[j]) == classes[i][int(class_label[j])]:
                counts[i] += 1
    return np.max(counts) / real_label.shape[0]


def uci_iris():
    data_set = pd.read_csv("./iris.csv")
    classes = data_set['class']
    X = np.zeros((data_set.shape[0], data_set.shape[1]-1))
    X[:, :] = np.array(data_set.drop('class', axis=1), dtype=float)
    label = np.zeros(data_set.shape[0])
    for i in range(classes.shape[0]):
        if classes[i] == 'Iris-setosa':
            continue
        elif classes[i] == 'Iris-versicolor':
            label[i] = 1
        elif classes[i] == 'Iris-virginica':
            label[i] = 2
    return X, label


train_x = generate_data(3, means=mean, covs=cov, sample_num=train_size)

_result, _label, _center = kmeans(train_x, 3)
showResult(_result, _label, _center)

# uci测试
test_x, real_label = uci_iris()

_result, _label, _center = kmeans(test_x, 3)
print(accuracy(real_label, _label, 3))
