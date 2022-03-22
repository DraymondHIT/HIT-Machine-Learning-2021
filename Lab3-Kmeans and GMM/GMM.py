import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import permutations

# 超参数
mean = [[4, 8], [10, 4], [10, 12]]
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
    dimension = X.shape[1]
    sample_size = X.shape[0]

    # 初始化中心点坐标和标签
    center = np.zeros((k, dimension))
    label = np.zeros(sample_size)

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


def Gaussian(x, mu, sigma):
    return 1 / ((2 * np.pi) * pow(np.linalg.det(sigma), 0.5)) * np.exp(
        -0.5 * (x - mu).dot(np.linalg.pinv(sigma)).dot((x - mu).T))


def E_step(X, alpha, mu, sigma):
    sample_size = X.shape[0]
    cluster_size = mu.shape[0]
    gamma = np.zeros((sample_size, cluster_size))
    p = np.zeros(cluster_size)
    p_x = np.zeros(cluster_size)
    for i in range(sample_size):
        for j in range(cluster_size):
            p[j] = Gaussian(X[i], mu[j], sigma[j])
            p_x[j] = alpha[j] * p[j]
        for j in range(cluster_size):
            gamma[i, j] = p_x[j] / np.sum(p_x)
    return gamma


def M_step(X, k, gamma):
    sample_size = X.shape[0]
    feature_size = X.shape[1]

    mu = np.zeros((k, feature_size))
    sigma = np.zeros((k, feature_size, feature_size))

    for i in range(k):
        # 计算新均值向量
        mu[i] = np.sum(X * gamma[:, i].reshape((-1, 1)), axis=0) / np.sum(gamma, axis=0)[i]

        # 计算新协方差矩阵
        sigma[i] = 0
        for j in range(sample_size):
            sigma[i] += (X[j].reshape((1, -1)) - mu[i]).T.dot((X[j] - mu[i]).reshape((1, -1))) * gamma[j, i]
        sigma[i] = sigma[i] / np.sum(gamma, axis=0)[i]

    # 计算新混合系数
    alpha = np.sum(gamma, axis=0) / sample_size

    return alpha, mu, sigma


def likelihood_calculate(X, alpha, mu, sigma):
    sample_size = X.shape[0]
    k = len(alpha)
    likelihood = 0

    for j in range(sample_size):
        temp = 0
        for i in range(k):
            temp += alpha[i] * Gaussian(X[j], mu[i], sigma[i])
        likelihood += np.log(temp)

    return likelihood


def GMM(data, k, epsilon=1e-5):
    epoch = 0
    sample_size = data.shape[0]
    feature_size = data.shape[1]

    # 初始化alpha, mu, sigma
    alpha = np.ones(k) / k
    _, _, mu = kmeans(data, k)  # 利用kmeans算法初始化簇中心坐标
    sigma = np.full((k, feature_size, feature_size), np.diag(np.full(feature_size, 0.1)))

    while True:
        epoch += 1
        last_alpha, last_mu, last_sigma = alpha, mu, sigma

        # E_step
        gamma = E_step(data, alpha, mu, sigma)

        # M_step
        alpha, mu, sigma = M_step(data, k, gamma)

        # 计算似然值
        likelihood = likelihood_calculate(data, alpha, mu, sigma)

        print("epoch {:>2d}: likelihood = {:+.10f}".format(epoch, likelihood))

        if np.linalg.norm(alpha - last_alpha) < epsilon and np.linalg.norm(mu - last_mu) < epsilon and np.linalg.norm(sigma - last_sigma) < epsilon:
            break

    label = np.argmax(gamma, axis=1)

    return alpha, mu, sigma, label


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

_alpha, _mu, _sigma, _label = GMM(train_x, 3)
showResult(train_x, _label, _mu)

# uci测试
test_x, real_label = uci_iris()

_alpha, _mu, _sigma, _label = GMM(test_x, 3)
print(accuracy(real_label, _label, 3))
