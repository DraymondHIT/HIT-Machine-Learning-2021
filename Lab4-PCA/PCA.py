import numpy as np
import cv2
from matplotlib import pyplot as plt

# 超参数
DIMENSION = 3
SAMPLE_SIZE = 100

FACE_NUM = 8
WIDTH = 100
HEIGHT = 100
CHANNEL = 1
REDUCED_DIMENSION = 1


# 生成数据
def generate_data(dimension, number):
    if dimension is 2:
        mean = [2, 3]
        cov = [[1, 0], [0, 0.01]]
    elif dimension is 3:
        mean = [1, 2, 3]
        cov = [[0.01, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        assert False
    sample_data = []
    for index in range(number):
        sample_data.append(np.random.multivariate_normal(mean, cov).tolist())
    return np.array(sample_data)


# 去中心化
def decentralise(data):
    return data - np.mean(data, axis=0)


# pca恢复
def pca_resume(data, feature_vectors):
    return decentralise(data).dot(feature_vectors).dot(feature_vectors.T) + np.mean(data, axis=0)


def pca(data, reduced_dimension):
    """
    主成分分析
    """
    data = np.float32(np.mat(data))
    # 数据去中心化
    decentralise_x = decentralise(data)
    # 计算协方差矩阵
    cov = np.cov(decentralise_x, rowvar=0)
    # 特征值分解
    eigenvalues, feature_vectors = np.linalg.eig(cov)
    # 选取最大的特征值对应的特征向量
    _min_index = np.argsort(eigenvalues)
    feature_vectors = feature_vectors[:, _min_index[-1:-(reduced_dimension + 1):-1, ]]

    return pca_resume(data, feature_vectors)


def psnr(source, target):
    MSE = np.mean(np.square(source-target))
    PSNR = 20*np.log10(255.0/np.sqrt(MSE))
    return PSNR


def draw_data(dimension, origin_data, pca_data):
    if dimension is 2:
        plt.scatter(origin_data[:, 0], origin_data[:, 1], color="purple", label="Origin Data")
        plt.scatter(pca_data[:, 0].tolist(), pca_data[:, 1].tolist(), color='orange', label='PCA Data')
    elif dimension is 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(origin_data[:, 0], origin_data[:, 1], origin_data[:, 2],
                   color="purple", label='Origin Data')
        ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], color='orange', label='PCA Data')
    else:
        assert False
    plt.legend()
    plt.show()


def load_faces(path, number):
    image = cv2.imread(path + str(number) + '.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    return image


def show_faces(data):
    for i in range(len(data)):
        plt.subplot(2, 4, i+1), plt.imshow(data[i]), plt.title(f'Image {i+1}')
        plt.axis('off')
    plt.show()


# 生成数据测试
# X = generate_data(DIMENSION, SAMPLE_SIZE)
# X_PCA = pca(X, DIMENSION-1)
# draw_data(DIMENSION, X, X_PCA)

# 人脸数据测试
path = './faces/'
_PCA_faces = []
faces = np.array(load_faces(path, 1)).reshape((-1, 1))
for i in range(1, FACE_NUM):
    face = np.array(load_faces(path, i+1)).reshape((-1, 1))
    faces = np.column_stack((faces, face))

PCA_faces = pca(faces, REDUCED_DIMENSION)
PCA_faces = np.real(PCA_faces)
for i in range(FACE_NUM):
    print(f"Image {i + 1} PSNR: {psnr(PCA_faces[:, i].reshape((HEIGHT, WIDTH)), faces[:, i].reshape((HEIGHT, WIDTH)))}")
    _PCA_faces.append(PCA_faces[:, i].reshape((HEIGHT, WIDTH)))
show_faces(np.array(_PCA_faces, dtype='uint8'))
