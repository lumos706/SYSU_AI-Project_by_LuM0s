import heapq
import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import MDS


def sklearn_KNN(train_array, test_array, train_message, test_message, k):
    # 创建KNN分类器
    knn = KNeighborsClassifier(n_neighbors=k)

    # 提取训练集和测试集的标签
    train_labels = [message[1] for message in train_message]
    test_labels = [message[1] for message in test_message]

    # 训练KNN分类器
    knn.fit(train_array, train_labels)

    # 使用KNN分类器进行预测
    predicted_labels = knn.predict(test_array)

    # 计算预测的准确率
    accuracy = accuracy_score(test_labels, predicted_labels)

    print("++++++++++++++++++++++++++++++++++++")
    print("\033[94mKNN Classifier from sklearn:\033[0m")
    print("\033[94mAccuracy:\033[0m {:.6f}%".format(accuracy * 100))
    print("++++++++++++++++++++++++++++++++++++")


def calculate_rmse(x_train, x_test, train_message, test_message):
    y_train = [message[1] for message in train_message]
    y_test = [message[1] for message in test_message]
    rmse_val = []  # to store rmse values for different k
    for K in range(100):
        K = K + 1
        model = neighbors.KNeighborsRegressor(n_neighbors=K)

        model.fit(x_train, y_train)  # fit the model
        pred = model.predict(x_test)  # make prediction on test set
        error = sqrt(mean_squared_error(y_test, pred))  # calculate rmse
        rmse_val.append(error)  # store rmse values
        # print('RMSE value for k= ', K, 'is:', error)
    # 绘制不同k值下的均方根损失
    curve = pd.DataFrame(rmse_val)  # elbow curve

    curve.plot()
    plt.title('Elbow curve')
    plt.xlabel('k')
    plt.xticks(np.arange(0, 100, step=4))
    plt.show(block=False)
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break


def plot_data(test_data, train_data, train_message, distance_func):
    # 创建一个映射，将情绪标签映射到颜色
    emotion_to_color = {1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y', 6: 'k'}
    """
    'b': 蓝色（blue）
    'g': 绿色（green）-> anger
    'r': 红色（red）-> disgust
    'c': 青色（cyan）-> fear
    'm': 品红色（magenta）-> joy
    'y': 黄色（yellow）-> sad
    'k': 黑色（black）-> surprise
    """
    pca = PCA(n_components=2)
    train_data_2d = pca.fit_transform(train_data)

    if len(test_data.shape) == 1:
        test_data = test_data.reshape(1, -1)
    test_data_2d = pca.transform(test_data)

    distances = []
    labels = []
    for i in range(len(train_data)):  # 遍历训练集
        dist = distance_func(test_data_2d[0], train_data_2d[i])  # 计算距离
        distances.append(dist)
        labels.append(train_message[i][1])
        # 使用映射来确定颜色
        plt.scatter(train_data_2d[i, 0], train_data_2d[i, 1], color=emotion_to_color[labels[i]], s=5)
    # 找出最近的15个点
    distances.sort()
    nearest_15 = distances[:15]

    # 画出包含最近的15个点的圆
    circle_radius = nearest_15[-1]  # 最远的那个最近点的距离，即圆的半径
    circle = plt.Circle((test_data_2d[0, 0], test_data_2d[0, 1]), circle_radius, fill=False)
    plt.gca().add_patch(circle)

    plt.scatter(test_data_2d[0, 0], test_data_2d[0, 1], color='blue', label='Test data', s=25)
    # 添加标题，标注使用的距离函数
    plt.title('Data Plot with Distance Function: {}'.format(distance_func.__name__))
    plt.show(block=False)
    while True:
        if plt.waitforbuttonpress():
            plt.close()
            break


def Euclidean_distance(X, Y):  # 欧氏距离
    return np.sqrt(np.sum((X - Y) ** 2))


def Manhattan_distance(X, Y):  # 曼哈顿距离
    return np.sum(abs(X - Y))


def Minkowski_distance(X, Y):  # 闵氏距离
    return pow((np.sum(abs(X - Y) ** (len(X)))), 1 / len(X))


def KNN_cosine_similarity(train, test, k, train_message, test_message):
    correct_num = 0
    time_start = time.time()
    n = len(test_message)
    similarity = cosine_similarity(test, train)  # 计算余弦相似度
    for i in range(n):  # 遍历测试集
        indices = np.argsort(similarity[i])[-k:]  # 选取相似度最大的k个训练集元素的索引（余弦相似度越大，向量越相似）
        emotion = np.zeros(6)
        distance = np.zeros(6)
        for j in range(k):
            emotion[train_message[indices[j]][1] - 1] += 1
            distance[train_message[indices[j]][1] - 1] += similarity[i][indices[j]]
        # 使用优先队列选取最多的类别
        queue = []
        for p in range(6):
            heapq.heappush(queue, (-emotion[p], -distance[p], p))
        _, _, max_id = heapq.heappop(queue)
        if max_id == test_message[i][1] - 1:
            correct_num += 1

    time_end = time.time()
    spend_time = time_end - time_start
    print("++++++++++++++++++++++++++++++++++++")
    print("\033[94mDistance metric:\033[0m Cosine_similarity")
    print("\033[94mCost time:\033[0m", spend_time, "s")
    print("\033[94mAccuracy:\033[0m {:.4f}%".format(correct_num / n * 100))
    print("++++++++++++++++++++++++++++++++++++")


def KNN_general(train_array, test_array, k, train_message, test_message, distance_func):
    correct_num = 0

    time_start = time.time()
    m = len(train_message)
    n = len(test_message)

    for i in range(n):  # 遍历测试集
        distance = []
        for j in range(m):  # 遍历训练集
            distance.append(distance_func(test_array[i], train_array[j]))  # 计算距离
        indices = np.argsort(distance)[:k]  # 选取距离最小的k个的索引
        emotion = np.zeros(6)  # 存放每个情绪类别的个数（0：anger 1：disgust 2：fear 3：joy 4：sad 5：surprise）
        distance_tmp = np.zeros(6)  # 存放每个情绪类别的距离
        for j in range(k):  # 统计k个最近邻的类别
            emotion[train_message[indices[j]][1] - 1] += 1  # 统计每个类别的个数
            distance_tmp[train_message[indices[j]][1] - 1] += distance[indices[j]]  # 统计每个类别的距离
        # 使用优先队列选取最多的类别
        queue = []
        for p in range(0, 6):
            heapq.heappush(queue, (-emotion[p], -distance_tmp[p], p))
        _, _, max_id = heapq.heappop(queue)
        if max_id == test_message[i][1] - 1:
            correct_num += 1

    time_end = time.time()
    spend_time = time_end - time_start
    print("++++++++++++++++++++++++++++++++++++")
    print("\033[94mDistance metric:\033[0m", distance_func.__name__)
    print("\033[94mCost time:\033[0m", spend_time, "s")
    print("\033[94mAccuracy:\033[0m {:.4f}%".format(correct_num / n * 100))
    print("++++++++++++++++++++++++++++++++++++")
    plot_data(test_array[1], train_array, train_message, distance_func)  # 绘制第一个测试集的数据


def read_file(f):
    message = []  # 存放对应的信息特征
    sentence = []  # 存放文本信息
    for line in f:
        tmp = line.strip().split(" ")
        if tmp[0] == "documentId":  # 跳过表头
            continue
        documentId = int(tmp[0])  # 文本序号
        emotionId = int(tmp[1])  # 情绪编号
        emotion = tmp[2]  # 分类标签
        single_sentence = ' '.join(tmp[3:])  # 提取每一句的单词
        sentence.append(single_sentence)  # 存放文本信息
        # 0：documentId, 1：emotionId, 2：emotion, 3：sentence
        # 其中，emotionId-1即为情绪类别（0：anger 1：disgust 2：fear 3：joy 4：sad 5：surprise）
        message.append([documentId, emotionId, emotion, single_sentence])
    return message, sentence


def main():
    distance_funcs = [Euclidean_distance, Manhattan_distance, Minkowski_distance]
    k = 16  # 选取k值(根号n)
    print("++++++++++++++++++++++++++++++++++++")
    print("\033[94mValue of k:\033[0m", k)

    f = open(r".\Classification\train.txt", 'r')
    train_message, train_sentence = read_file(f)  # 读训练集
    f = open(r".\Classification\test.txt", 'r')
    test_message, test_sentence = read_file(f)  # 读测试集

    # TF-IDF 提取文本特征
    t = TfidfVectorizer()
    train = t.fit_transform(train_sentence)  # 读取训练集特征，此时返回一个sparse矩阵
    test = t.transform(test_sentence)  # 读取测试集特征，此时返回一个sparse矩阵

    train_array = np.array(train.toarray())
    test_array = np.array(test.toarray())
    # print("\033[94mTest_array[0]:\033[0m", test_array[0])

    calculate_rmse(train_array, test_array, train_message, test_message)  # 计算不同k值下的均方根损失
    # 打印train和test的基本信息
    print("\033[94mTrain data shape:\033[0m", train.shape)
    print("\033[94mTest data shape:\033[0m", test.shape)
    print("++++++++++++++++++++++++++++++++++++")

    KNN_cosine_similarity(train, test, k, train_message, test_message)  # 使用余弦相似度进行knn分类
    for distance_func in distance_funcs:
        KNN_general(train_array, test_array, k, train_message, test_message, distance_func)  # 使用其他距离进行knn分类
    sklearn_KNN(train_array, test_array, train_message, test_message, k)  # 使用sklearn中的KNN分类器


if __name__ == '__main__':
    main()
