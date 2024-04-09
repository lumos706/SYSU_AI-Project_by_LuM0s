import copy
import time
from queue import PriorityQueue


# 定义一个表示拼图状态的类
class Puzzle:
    """
    Puzzle类用于表示一个拼图的状态。
    属性:
        state: 一个元组，表示拼图的当前状态。
        path: 一个列表，表示从初始状态到当前状态的路径。
        g: 一个整数，表示从初始状态到当前状态的路径的成本。
        h: 一个整数，表示当前状态的启发式值。
        zero: 一个整数，表示空白格在拼图中的索引。
    """

    def __init__(self, list16):
        """
        初始化Puzzle类的一个新实例。
        参数:
            list16: 一个列表，包含16个整数，表示拼图的初始状态。
            path: 一个列表，表示从初始状态到当前状态的路径。
            g: 一个整数，表示从初始状态到当前状态的路径的成本。
            h: 一个整数，表示当前状态的启发式值。
            zero: 一个整数，表示空白格在拼图中的索引。
        """
        self.state = tuple(list16)  # 使用元组以便内存减少，所以状态应该是不可变的
        self.path = []  # 初始化路径为空
        self.g = 0
        self.h = 0
        self.zero = self.state.index(0)  # 找到空白格的索引
        self.hn()  # 计算启发式值

    def __lt__(self, other, h_mul=1.0, g_mul=1.0):
        """
        定义两个Puzzle实例之间的比较方法。
        参数:
            other: 另一个Puzzle实例。
            h_mul: 启发式值的权重。
            g_mul: 路径成本的权重。

        返回:
            如果当前实例的f值小于另一个实例的f值，或者f值相等但h值小于另一个实例的h值，返回True，否则返回False。
        """
        return g_mul * self.g + h_mul * self.h < g_mul * other.g + h_mul * other.h or (
                g_mul * self.g + h_mul * self.h == g_mul * other.g + h_mul * other.h and self.h < other.h)

    def hn(self):
        """
        计算当前状态的启发式值。
        """
        h1 = 0
        h2 = 0
        for i in range(4):
            for j in range(4):
                num = self.state[i * 4 + j]
                if num != 0:
                    row_goal = (num - 1) // 4
                    col_goal = (num - 1) % 4
                    h1 += abs(i - row_goal) + abs(j - col_goal)
                    if j == col_goal:
                        for k in range(j + 1, 4):
                            next_num = self.state[i * 4 + k]
                            if next_num != 0 and (next_num - 1) // 4 == i and (next_num - 1) % 4 < j:
                                h2 += 1
                    if i == row_goal:
                        for k in range(i + 1, 4):
                            next_num = self.state[k * 4 + j]
                            if next_num != 0 and (next_num - 1) % 4 == j and (next_num - 1) // 4 < i:
                                h2 += 1

        self.h = h1 + h2

    def successors(self, visited):
        """
        生成当前状态的所有后继状态。
        返回:
            一个生成器，生成所有后继状态的Puzzle实例。
        """
        zero = self.state.index(0)
        x = zero % 4
        y = zero // 4
        move = []
        if x > 0: move.append(-1)
        if y > 0: move.append(-4)
        if x < 3: move.append(1)
        if y < 3: move.append(4)
        for i in range(len(move)):
            new = list(self.state)
            new[zero], new[zero + move[i]] = new[zero + move[i]], new[zero]  # swap the blank and the number
            if tuple(new) in visited:
                continue
            puz_child = Puzzle(new)
            puz_child.g = self.g + 1
            puz_child.path = self.path[:]
            puz_child.path.append(new[zero])
            yield puz_child


def A_star(p):
    """
    使用A*算法解决拼图问题。
    参数:
        p: 一个Puzzle实例，表示拼图的初始状态。
    返回:
        一个Puzzle实例，表示拼图的解决方案。
    """
    sum = 0
    pri = PriorityQueue()  # 创建一个优先队列
    pri.put(p)  # 将初始状态放入队列
    visited = set()  # 创建一个集合来存储已访问过的状态
    while not pri.empty():  # 当队列不为空时
        puz = pri.get()  # 从队列中获取一个状态
        if puz.state in visited:  # 如果这个状态已经被访问过，就跳过
            continue
        sum += 1
        if puz.h == 0:  # 如果启发式值为0，说明找到了解决方案
            print("Finding", sum, "nodes")
            return puz
        visited.add(puz.state)  # 将当前状态添加到已访问集合中
        for puz_child in puz.successors(visited):  # 遍历当前状态的所有后继状态
            pri.put(puz_child)  # 将后继状态添加到队列中


def IDA_star(p):
    """
    使用IDA*算法解决拼图问题。
    参数:
        p: 一个Puzzle实例，表示拼图的初始状态。
    返回:
        一个列表，包含从初始状态到解决方案的所有状态的Puzzle实例。
    """
    threshold_value = p.h  # 设置初始阈值为启发式值
    visited = set()  # 创建一个集合来存储已访问过的状态
    visited.add(p.state)  # 记录初始状态
    while True:
        threshold_value, newstate = search(p, 0, threshold_value, visited)
        if threshold_value == 'FOUND':  # 如果找到了解决方案，返回解决方案
            return newstate
        if threshold_value == float('inf'):  # 如果阈值为无穷大，说明没有解决方案
            return p


def search(state, g, limit, visited):
    """
    使用深度优先搜索解决拼图问题。
    参数:
        state: 一个Puzzle实例，表示当前状态。
        g: 一个整数，表示当前路径的成本。
        limit: 一个整数，表示搜索的深度限制。
        visited: 一个集合，包含已经访问过的所有状态。
    返回:
        如果找到解决方案，返回 'FOUND'；否则返回下一次搜索的深度限制。
    """
    f = g + state.h  # 计算f值
    if f > limit:  # 如果f值大于阈值，返回f值和当前状态
        return f, state
    if state.h == 0:  # 如果启发式值为0，说明找到了解决方案
        return 'FOUND', state
    min_val = float('inf')  # 初始化最小值为无穷大
    for successor in state.successors(visited):  # 遍历当前状态的所有后继状态
        visited.add(successor.state)  # 将后继状态添加到已访问集合中
        temp, newstate = search(successor, g + 1, limit, visited)  # 递归搜索后继状态
        if temp == 'FOUND':  # 如果找到了解决方案，返回'FOUND'和解决方案
            return 'FOUND', newstate
        if temp < min_val:  # 如果新的f值小于最小值，更新最小值
            min_val = temp
        visited.remove(successor.state)  # 回溯，从已访问集合中移除当前状态
    return min_val, state  # 返回最小值和当前状态


def print_puzzle(matrix):
    """
    打印拼图的状态。
    参数:
        matrix: 一个列表，包含16个整数，表示拼图的状态。
    """
    for i in range(4):
        print(" ".join(str(matrix[4 * i + j]) for j in range(4)))


def move_blank(matrix, move):
    """
    移动拼图中的空白格。
    参数:
        matrix: 一个列表，包含16个整数，表示拼图的状态。
        move: 一个整数，表示要与空白格交换位置的数字。
    """
    zero_index = matrix.index(0)
    move_index = matrix.index(move)
    matrix[zero_index], matrix[move_index] = matrix[move_index], matrix[zero_index]


def main():
    """
    主函数，从文件中读取拼图的初始状态，然后使用A*算法和IDA*算法解决拼图问题，并打印解决方案和运行时间。
    """
    # matrix = [0, 5, 15, 14, 7, 9, 6, 13, 1, 2, 12, 10, 8, 11, 4, 3]  # example initial state
    with open('input7.txt', 'r') as f:
        lines = f.readlines()
    matrix = [int(num) for line in lines for num in line.split()]
    matrix1 = copy.deepcopy(matrix)
    # print(matrix)
    a = Puzzle(matrix)
    time_start = time.time()  # 计算时间
    ans = A_star(a)
    time_end = time.time()
    time_c = time_end - time_start

    print("A* Algorithm Solution:")
    print("Total moves:", ans.g)
    print('A*:', 'time cost', time_c, 's')
    print(ans.path, end='\nstate:\n')
    for move in ans.path:  # 打印解决方案
        print_puzzle(matrix1)
        print("----------")
        move_blank(matrix1, move)
    print_puzzle(matrix1)
    print("----------")
    # print(ans.path)

    time_start = time.time()  # 计算时间
    ans = IDA_star(a)
    time_end = time.time()
    time_c = time_end - time_start

    print("\nIDA* Algorithm Solution:")
    print("Total moves:", ans.g)
    print('IDA*:', 'time cost', time_c, 's')
    print(ans.path, end='\nstate:\n')
    for move in ans.path:  # 打印解决方案
        print_puzzle(matrix)
        print("----------")
        move_blank(matrix, move)
    print_puzzle(matrix)
    print("----------")


if __name__ == '__main__':
    main()
