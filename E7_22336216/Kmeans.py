import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['OMP_NUM_THREADS'] = '2'


def main():
    # 读取数据集
    data = pd.read_csv('kmeans_data.csv')
    # 创建KMeans聚类器
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, verbose=0,
                    random_state=None, copy_x=True, algorithm='lloyd')
    # 训练聚类器
    kmeans.fit(data)
    # 获取聚类结果
    labels = kmeans.labels_
    # 获取聚类中心
    centers = kmeans.cluster_centers_
    # 创建颜色和标记的列表
    colors = ['lightgreen', 'orange', 'lightblue']
    markers = ['s', 'o', 'v']
    # 为每个聚类创建一个scatter图
    for i in range(3):
        plt.scatter(data.iloc[labels == i, 0], data.iloc[labels == i, 1], s=25, c=colors[i], marker=markers[i], label=f'cluster {i+1}')
    # 创建一个scatter图来表示聚类的中心
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c='red', marker='*', label='centroids')
    plt.legend()
    plt.grid()
    plt.show()

    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
        km.fit(data)
        distortions.append(km.inertia_)

    plt.plot(range(1, 11), distortions, marker='o')
    plt.title('Elbow method')
    plt.xticks(range(1, 11))
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


if __name__ == '__main__':
    main()