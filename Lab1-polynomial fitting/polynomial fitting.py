import numpy as np
import matplotlib.pyplot as plt

loss_history = []

# 数据点个数
N = 10

# 多项式阶数
M = 3

# 标准曲线
x0 = np.linspace(0, 1, 100)
y0 = np.sin(2 * np.pi * x0)


# 采样函数
def generate_data(N):
    x = np.linspace(0, 1, N)
    y = np.sin(2 * np.pi * x) + np.random.normal(loc=0, scale=0.2, size=N)  # 增加高斯噪声
    return x.reshape(N, 1), y.reshape(N, 1)


# 最小二乘法
def regress(M, x, y, lamda=0):
    # 计算X
    order = np.arange(M + 1).reshape((1, -1))
    X = x ** order

    # 计算W
    W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) +
                                    lamda * np.identity(M + 1)), X.T), y)
    # loss = np.sum(0.5 * np.dot((y - np.dot(X, W)).T, y - np.dot(X, W)) + 0.5 * lamda * np.dot(W.T, W))
    X0 = x0.reshape(-1, 1) ** order
    return np.dot(X0, W), W


# 梯度下降
def gradient_descent(M, x, y, lr=0.01, delta=1e-6):
    # 初始化参数
    iter = 0
    W = np.ones(M + 1).reshape(-1, 1)
    order = np.arange(M + 1).reshape((1, -1))
    X = x ** order

    # 计算初始损失
    last_loss = np.sum(0.5 * np.dot((y - np.dot(X, W)).T, y - np.dot(X, W)))
    loss_history.append(last_loss)
    while True:
        # 计算梯度
        dW = np.dot(np.dot(X.T, X), W) - np.dot(X.T, y)

        # 梯度下降
        W -= lr * dW

        # 计算新的损失
        loss = np.sum(0.5 * np.dot((y - np.dot(X, W)).T, y - np.dot(X, W)))
        loss_history.append(loss)

        if np.abs(loss - last_loss) < delta:
            break
        else:
            # 更新迭代次数，更新损失
            iter += 1
            last_loss = loss

    X0 = x0.reshape(-1, 1) ** order
    return np.dot(X0, W), W, iter


# 共轭梯度下降法
def conjugate_gradient(M, x, y):
    # 初始化参数
    iter = 0
    W = np.ones(M + 1).reshape(-1, 1)
    order = np.arange(M + 1).reshape((1, -1))
    X = x ** order

    A = np.dot(X.T, X)
    b = np.dot(X.T, y)
    r = b - np.dot(A, W)
    p = r
    while True:
        # 迭代
        r1 = r
        alpha = np.dot(r.T, r) / np.dot(p.T, np.dot(A, p))
        W = W + alpha * p
        r = b - np.dot(A, W)

        # 计算误差
        q = np.linalg.norm(np.dot(A, W) - b) / np.linalg.norm(b)
        if q < 10 ** -6:
            break
        else:
            # 更新
            iter += 1
            beta = np.linalg.norm(r) ** 2 / np.linalg.norm(r1) ** 2
            p = r + beta * p

    X0 = x0.reshape(-1, 1) ** order
    return np.dot(X0, W), W, iter


# 获取带噪声的数据
x, y = generate_data(N)

# 四种方式获取y和W
y_1, W1 = regress(M, x, y)
y_2, W2 = regress(M, x, y, lamda=1e-4)
y_3, W3, iter = gradient_descent(M, x, y)
y_4, W4, iter = conjugate_gradient(M, x, y)

print(W1)
print(W2)
print(W3)
print(W4)

# 作图
plt.figure(1, figsize=(8, 5))
plt.plot(x0, y_1, 'orange', linewidth=2, label='without_penalty')
plt.plot(x0, y_2, 'r', linewidth=2, label='with_penalty')
plt.plot(x0, y_3, 'purple', linewidth=2, label='gradient_descent')
plt.plot(x0, y_4, 'pink', linewidth=2, label='conjugate_gradient')
plt.plot(x0, y0, 'b', linewidth=2, label='base')
plt.scatter(x, y, marker='o', edgecolors='b', s=100, linewidth=3)
plt.title(f'M = {M}, traing_num = {N}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="best", fontsize=10)
plt.show()
