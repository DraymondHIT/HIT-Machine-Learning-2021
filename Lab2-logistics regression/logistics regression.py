import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 超参数
mean_1 = [-1., -1.]
mean_2 = [1., 1.]
var = 0.4
cov = 0.1
train_size = 50
test_size = 200

loss_history = []


# 随机生成数据
def generate_data(mean_1, var_1, size_1, mean_2, var_2, size_2, cov=0.0):
    train_x = np.zeros((size_1 + size_2, 2))
    train_y = np.zeros(size_1 + size_2)
    train_x[:size_1, :] = np.random.multivariate_normal(
        mean=mean_1, cov=[[var_1, cov], [cov, var_1]], size=size_1)
    train_x[size_1:, :] = np.random.multivariate_normal(
        mean=mean_2, cov=[[var_2, cov], [cov, var_2]], size=size_2)
    train_y[size_1:] = 1
    return train_x.T, train_y.reshape(1, -1)


# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 损失函数
def calculate_loss(X, Y, W, lambd=0):
    size = Y.shape[1]
    loss = (np.sum(Y * W.T.dot(X)) - np.sum(np.log(1 + np.exp(W.T.dot(X)))) - 0.5 * lambd * W.T.dot(W)) / size
    return -float(loss)


# 梯度下降
def gradient_descent(X, Y, lambd=0, lr=0.05, epsilon=1e-6):
    # 初始化
    iter = 0
    size = X.shape[1]
    dimension = X.shape[0]
    ones = np.ones((1, size))
    X = np.row_stack((ones, X))  # 构造X的增广矩阵，增加一个全1的行
    W = np.ones((dimension + 1, 1))

    last_loss = calculate_loss(X, Y, W, lambd=lambd)

    while True:
        loss_history.append(last_loss)

        # 计算梯度
        dW = - np.sum(X * (Y - sigmoid(W.T.dot(X))), axis=1).reshape(-1, 1) + lambd * W
        dW /= size

        # 梯度下降
        W -= lr * dW

        # 计算新的损失
        loss = calculate_loss(X, Y, W, lambd=lambd)
        print(loss)

        if np.abs(loss - last_loss) < epsilon and np.dot(dW.T, dW) < epsilon:
            break
        else:
            # 更新迭代次数，更新损失
            iter += 1
            last_loss = loss

    coefficient = - W[:dimension, 0] / W[dimension]

    print(iter)

    return coefficient, W


# 海森阵
def Hessian(X, W, lambd=0):
    size = X.shape[1]
    dimension = W.shape[0]
    return (sigmoid(W.T.dot(X)) * sigmoid(-W.T.dot(X)) * X).dot(X.T) / size


# 牛顿法
def newton(X, Y, lambd=0, epsilon=1e-6):
    # 初始化
    iter = 0
    size = X.shape[1]
    dimension = X.shape[0]
    ones = np.ones((1, size))
    X = np.row_stack((ones, X))  # 构造X的增广矩阵，增加一个全1的行
    W = np.ones((dimension + 1, 1))

    last_loss = calculate_loss(X, Y, W, lambd=lambd)

    while True:
        loss_history.append(last_loss)

        # 计算梯度
        dW = - np.sum(X * (Y - sigmoid(W.T.dot(X))), axis=1).reshape(-1, 1) + lambd * W
        dW /= size

        # 计算海森阵
        H = Hessian(X, W, lambd=lambd)

        # 迭代
        W = W - np.linalg.inv(H).dot(dW)

        # 计算新的损失
        loss = calculate_loss(X, Y, W, lambd=lambd)
        # print(loss)

        if np.abs(loss - last_loss) < epsilon and np.dot(dW.T, dW) < epsilon:
            break
        else:
            # 更新迭代次数，更新损失
            iter += 1
            last_loss = loss

    coefficient = - W[:dimension, 0] / W[dimension]

    return coefficient, W


# 计算分类准确率
def accuracy(X, Y, W):
    total = X.shape[1]
    correct_num = 0

    ones = np.ones((1, total))
    X = np.row_stack((ones, X))

    for i in range(total):
        if sigmoid(W.T.dot(X[:, i])) > 0.5 and Y[0, i] == 1 or sigmoid(W.T.dot(X[:, i])) < 0.5 and Y[0, i] == 0:
            correct_num += 1

    return float(correct_num) / total


# 作图
def show_data(X, Y, coefficient1, coefficient2, coefficient3, title):
    X = X.T
    Y = Y.T
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, marker='o', cmap=plt.cm.Spectral)

    bottom = np.min(X[:, 0])
    top = np.max(X[:, 0])
    X = np.linspace(bottom, top, 100).reshape(-1, 1)
    Y = coefficient1[0] + coefficient1[1] * X
    plt.plot(X, Y, 'orange', linewidth=2, label='NW')

    Y = coefficient2[0] + coefficient2[1] * X
    plt.plot(X, Y, 'r', linewidth=2, label='GD')

    Y = coefficient3[0] + coefficient3[1] * X
    plt.plot(X, Y, 'b', linewidth=2, label='GDR')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc="best", fontsize=10)
    plt.title(title)
    plt.show()


# 损失曲线
def loss_curve(loss_history):
    plt.plot(np.linspace(1, len(loss_history) + 1, len(loss_history)), loss_history)
    plt.show()


# 读取UCI数据
def uci_data(path):
    data = np.loadtxt(path, dtype=np.int32)
    np.random.shuffle(data) # 随机打乱数据，便于选取数据
    dimension = data.shape[1]
    train_size = int(0.3 * data.shape[0]) # 按照3：7的比例分配训练集和测试集

    # 划分训练集和测试集
    train_data = data[:train_size, :]
    test_data = data[train_size:, :]

    train_x = train_data[:, 0:dimension-1]
    train_y = train_data[:, dimension-1] - 1
    test_x = test_data[:, 0:dimension-1]
    test_y = test_data[:, dimension-1] - 1

    return train_x.T, train_y.reshape(1, -1), test_x.T, test_y.reshape(1, -1)


# 绘制三维图像
def show_3D(X, Y, coefficient, title):
    X = X.T
    Y = Y.T
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.Spectral)
    real_x = np.linspace(np.min(X[:, 0]) - 20, np.max(X[:, 0]) + 20, 255)
    real_y = np.linspace(np.min(X[:, 1]) - 20, np.max(X[:, 1]) + 20, 255)
    real_X, real_Y = np.meshgrid(real_x, real_y)
    real_z = coefficient[0] + coefficient[1] * real_X + coefficient[2] * real_Y
    ax.plot_surface(real_x, real_y, real_z, rstride=1, cstride=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

# 生成训练样本
train_x, train_y = generate_data(mean_1, var, train_size, mean_2, var, train_size, cov)

# 牛顿法
coefficient1, W1 = newton(train_x, train_y)

# 梯度下降法
coefficient2, W2 = gradient_descent(train_x, train_y, lambd=0)
coefficient3, W3 = gradient_descent(train_x, train_y, lambd=1e-2)

# 计算训练集分类准确率
print('---------------train---------------')
print('newton_train', accuracy(train_x, train_y, W1))
print('gd_train', accuracy(train_x, train_y, W2))
print('gdr_train', accuracy(train_x, train_y, W3))

# 训练集分类结果
show_data(train_x, train_y, coefficient1, coefficient2, coefficient3, 'Train_set')


# 损失函数曲线
# loss_curve(loss_history)

# 生成测试样本
test_x, test_y = generate_data(mean_1, var, test_size, mean_2, var, test_size, cov)

# 计算测试集分类准确率
print('---------------test---------------')
print('newton_test', accuracy(test_x, test_y, W1))
print('gd_test', accuracy(test_x, test_y, W2))
print('gdr_test', accuracy(test_x, test_y, W3))

# 测试集分类结果
show_data(test_x, test_y, coefficient1, coefficient2, coefficient3, 'Test_set')

# UCI数据集测试
# train_x, train_y, test_x, test_y = uci_data('haberman.txt')
# coefficient, W = gradient_descent(train_x, train_y, lr=0.0005, lambd=1e-3, epsilon=1e-5)
# print('uci_train', accuracy(train_x, train_y, W))
# show_3D(train_x, train_y, coefficient, 'Train_set')
#
# print('uci_test', accuracy(test_x, test_y, W))
# show_3D(test_x, test_y, coefficient, 'Test_set')
