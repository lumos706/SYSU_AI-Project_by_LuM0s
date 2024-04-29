import time
from math import floor

import matplotlib.pyplot as plt
import numpy as np


def read_tsp_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 初始化一个空列表来存储坐标
    coords = []

    # 标志位，用于指示我们是否在文件的节点部分
    is_node_section = False

    for line in lines:
        # 检查我们是否已经到达节点部分
        if line.strip() == 'NODE_COORD_SECTION':
            is_node_section = True
            continue

        # 检查我们是否已经到达文件的结尾
        if line.strip() == 'EOF':
            break

        # 如果我们在节点部分，解析行以获取坐标
        if is_node_section:
            parts = line.strip().split()
            if len(parts) >= 3:  # 确保我们有足够的部分（x和y坐标）
                x_coord = float(parts[1])
                y_coord = float(parts[2])
                coords.append((x_coord, y_coord))

    # 将坐标列表转换为numpy数组
    cities = np.array(coords)

    return cities


class GeneticAlgTSP(object):
    def __init__(self, cities, Max=1000, size_pop=100, cross_prob=0.80, mutation_prob=0.02, select_prob=0.8):
        self.Max = Max  # 最大迭代次数
        self.cross_prob = cross_prob  # 交叉概率
        self.mutation_prob = mutation_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率
        self.size_pop = size_pop  # 群体个数
        '''
        # 打开文件并读取内容  
  
  
        '''
        self.cities = cities

        # self.cities.append(list(map(float, line.strip().split()[1:])))
        self.num = len(cities)  # 城市个数，对应染色体长度
        self.matrix_distance = self.distance()
        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)  # 通过选择概率确定子代的选择个数
        self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop, self.num)  # 父种群
        self.sub_sel = np.array([0] * int(self.select_num) * self.num).reshape(self.select_num, self.num)  # 子种群
        self.fitness = np.zeros(self.size_pop)
        self.best_fit = []  # 最优距离
        self.best_path = []  # 最优路径

    def distance(self):  # 计算两个城市间的距离
        matrix = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in range(i + 1, self.num):
                matrix[i, j] = np.linalg.norm(self.cities[i, :] - self.cities[j, :])
                matrix[j, i] = matrix[i, j]
        return matrix

    def rand_chrom(self):  # 随机产生初始化种群
        population = np.array((range(self.num)))  # num=29
        for i in range(self.size_pop):  # size_pop为群体个数
            np.random.shuffle(population)  # 打乱城市染色体编码
            self.chrom[i, :] = population
            self.fitness[i] = self.comp_fit(population)

    def comp_fit(self, path):  # 计算单个染色体的路径距离值
        res = 0
        for i in range(self.num - 1):
            res += self.matrix_distance[path[i], path[i + 1]]  # [i,j]表示城市i到城市j的距离
        res += self.matrix_distance[path[-1], path[0]]  # 加上最后一个城市到起点的距离
        return res

    def out_path(self, path):  # 显示出路径
        res = str(path[0] + 1) + '-->'
        for i in range(1, self.num):
            res += str(path[i] + 1) + '-->'
        res += str(path[0] + 1) + '\n'
        print(res)

    def select_sub(self):  # 选取子代的函数,采用随机遍历选择法
        fit = 1. / self.fitness
        cumsum_fit = np.cumsum(fit)  # 求和,计算轴向的累加和,本行 = 本行 + 上一行
        pick = cumsum_fit[-1] / self.select_num * (
                np.random.rand() + np.array(range(int(self.select_num))))  # select_num为子代选择个数,pick为选择的片段
        i = 0
        j = 0
        index = []
        while i < self.size_pop and j < self.select_num:
            if cumsum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index, :]

    def cross_sub(self):  # 交叉，对子代染色体之间进行交叉操作
        if self.select_num % 2 == 0:
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num - 1), 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i, :], self.sub_sel[i + 1, :] = self.intercross(self.sub_sel[i, :], self.sub_sel[i + 1, :])

    def intercross(self, r_a, r_b):  # r_a,r_b为两条父代染色体
        r1 = np.random.randint(self.num)  # 在城市个数范围内随机生成一个数,此处生成的是交换区间
        r2 = np.random.randint(self.num)
        while r1 == r2:
            r2 = np.random.randint(self.num)  # 确保两个随机生成的数不相等
        if r1 > r2:
            right = r1
            left = r2
        else:
            right = r2
            left = r1
        r_a1 = r_a.copy()
        r_b1 = r_b.copy()
        for i in range(left, right + 1):  # left<=i<right+1
            r_a2 = r_a.copy()
            r_b2 = r_b.copy()
            r_a[i] = r_b1[i]  # 该步骤是依据随机生成的区间进行两条染色体的交叉操作
            r_b[i] = r_a1[i]
            x = np.argwhere(r_a == r_a[i])  # 返回数组里面大于某个设定值的数对应的索引
            y = np.argwhere(r_b == r_b[i])
            if len(x) == 2:
                r_a[x[x != i]] = r_a2[i]
            if len(y) == 2:
                r_b[y[y != i]] = r_b2[i]  # 首先判断交叉后是否有重复的元素，如果有是不满足条件的，要将交叉后重复的元素改回来

        return r_a, r_b

    def Mutation(self):  # 变异
        for i in range(int(self.select_num)):  # 遍历每一个选择的子代
            if np.random.rand() <= self.cross_prob:  # 以变异概率进行变异
                r1 = np.random.randint(self.num)
                r2 = np.random.randint(self.num)
                while r1 == r2:
                    r2 = np.random.randint(self.num)  # 确保随机生成的两个数不相等
                self.sub_sel[i, [r1, r2]] = self.sub_sel[i, [r2, r1]]  # 随机交换两个点的位置

    def reverse_sub(self):  # 进化逆转
        for i in range(int(self.select_num)):
            r1 = np.random.randint(self.num)
            r2 = np.random.randint(self.num)
            while r1 == r2:
                r2 = np.random.randint(self.num)  # 确保随机生成的两个数不相等
            if r1 > r2:
                right = r1
                left = r2
            else:
                right = r2
                left = r1
            father = self.sub_sel[i, :].copy()  # father为父代染色体
            father[left:right + 1] = self.sub_sel[i, left:right + 1][::-1]  # 逆转操作
            if self.comp_fit(father) < self.comp_fit((self.sub_sel[i, :])):  # 检查和判断翻转后的染色体和原来的染色体的适应度
                self.sub_sel[i, :] = father

    def rein(self):  # 将子代重新插入父代
        index = np.argsort(self.fitness)[::-1]  # 此处为倒序，将最差的替换掉
        self.chrom[index[:self.select_num], :] = self.sub_sel


# cities = np.array([20833.3333, 17100.0000, 20900.0000, 17066.6667,
#                    21300.0000, 13016.6667, 21600.0000, 14150.0000,
#                    21600.0000, 14966.6667, 21600.0000, 16500.0000,
#                    22183.3333, 13133.3333, 22583.3333, 14300.0000,
#                    22683.3333, 12716.6667, 23616.6667, 15866.6667,
#                    23700.0000, 15933.3333, 23883.3333, 14533.3333,
#                    24166.6667, 13250.0000, 25149.1667, 12365.8333,
#                    26133.3333, 14500.0000, 26150.0000, 10550.0000,
#                    26283.3333, 12766.6667, 26433.3333, 13433.3333,
#                    26550.0000, 13850.0000, 26733.3333, 11683.3333,
#                    27026.1111, 13051.9444, 27096.1111, 13415.8333,
#                    27153.6111, 13203.3333, 27166.6667, 9833.3333,
#                    27233.3333, 10450.0000, 27233.3333, 11783.3333,
#                    27266.6667, 10383.3333, 27433.3333, 12400.0000,
#                    27462.5000, 12992.2222]).reshape(29, 2)


def main(cities):
    path = GeneticAlgTSP(cities)
    path.rand_chrom()

    # 绘制初始化的路径图
    fig, ax = plt.subplots()
    x = cities[:, 0]
    y = cities[:, 1]
    ax.scatter(x, y, linewidths=0.1)
    for i, txt in enumerate(range(1, 29 + 1)):
        ax.annotate(txt, (x[i], y[i]))
    res0 = path.chrom[0]
    x0 = x[res0]
    y0 = y[res0]
    for i in range(len(cities) - 1):
        plt.quiver(x0[i], y0[i], x0[i + 1] - x0[i], y0[i + 1] - y0[i], color='r', width=0.002, angles='xy', scale=1,
                   scale_units='xy')
    plt.quiver(x0[-1], y0[-1], x0[0] - x0[-1], y0[0] - y0[-1], color='r', width=0.002, angles='xy', scale=1,
               scale_units='xy')
    plt.show(block=False)
    plt.waitforbuttonpress()  # 等待用户按键
    plt.close()  # 关闭图形
    print('初始染色体的路程: ' + str(path.fitness[0]))

    for i in range(path.Max):
        path.select_sub()
        path.cross_sub()
        path.Mutation()
        path.reverse_sub()
        path.rein()
        for j in range(path.size_pop):
            path.fitness[j] = path.comp_fit(path.chrom[j, :])  # 重新计算群体的距离值

        index = path.fitness.argmin()  # 求数组最小值索引
        if (i + 1) % 50 == 0:
            timestamp = time.time()
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))  # ???
            print(formatted_time)
            print('第' + str(i + 1) + '代后最短路径为：' + str(path.fitness[index]))
            print('第' + str(i + 1) + '代后最优路径为：')
            path.out_path(path.chrom[index, :])

        path.best_fit.append(path.fitness[index])
        path.best_path.append(path.chrom[index, :])  # 存储每一步的最优路径和距离
    res1 = path.chrom[0]
    x0 = x[res1]
    y0 = y[res1]
    for i in range(29 - 1):
        plt.quiver(x0[i], y0[i], x0[i + 1] - x0[i], y0[i + 1] - y0[i], color='r', width=0.002, angles='xy', scale=1,
                   scale_units='xy')
    plt.quiver(x0[-1], y0[-1], x0[0] - x0[-1], y0[0] - y0[-1], color='r', width=0.002, angles='xy', scale=1,
               scale_units='xy')
    plt.show(block=False)
    plt.waitforbuttonpress()  # 等待用户按键
    plt.close()  # 关闭图形
    return path


if __name__ == '__main__':
    cities = read_tsp_data("C:\\Users\\10559\\Desktop\\学习\\大二\\人工智能\\AI by taoyzh\\E0_22336216_TSP\\wi29.tsp")
    path = main(cities)
    print(path)
