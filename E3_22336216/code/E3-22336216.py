import copy
import time
from queue import PriorityQueue


class Puzzle:
    def __init__(self, list16):
        """
        Initialize the puzzle.
        state: the state of the puzzle
        path: the path of the puzzle
        g: the cost of the path
        h: the heuristic value of the puzzle
        zero: the index of the blank
        hn: calculate the heuristic value of the puzzle
        __lt__: compare in priority queue
        successors: generate the successors of the puzzle
        """
        self.state = tuple(list16)  # using tuple to hash, so the state should be immutable
        self.path = []
        self.g = 0
        self.h = 0
        self.zero = self.state.index(0)
        self.hn()

    def __lt__(self, other, h_mul=1.0, g_mul=1.0):
        """
        Compare in priority queue.
        """
        return g_mul * self.g + h_mul * self.h < g_mul * other.g + h_mul * other.h or (
                g_mul * self.g + h_mul * self.h == g_mul * other.g + h_mul * other.h and self.h < other.h)

    # def hn(self):
    #     """
    #     Calculate the heuristic value of the puzzle.
    #     using Manhattan distance.
    #     """
    #     self.h = sum(
    #         abs((self.state[i] - 1) % 4 - i % 4) + abs(int(((self.state[i]) - 1) / 4) - i // 4) for i in range(16) if
    #         self.state[i] != 0)
    def hn(self):
        h1 = sum(
            abs((self.state[i] - 1) % 4 - i % 4) + abs(int(((self.state[i]) - 1) / 4) - i // 4) for i in range(16) if
            self.state[i] != 0)

        h2 = 0
        for i in range(4):
            for j in range(4):
                if self.state[i * 4 + j] != 0:
                    row_goal = (self.state[i * 4 + j] - 1) // 4
                    col_goal = (self.state[i * 4 + j] - 1) % 4

                    if j == col_goal:
                        for k in range(j + 1, 4):
                            if self.state[i * 4 + k] != 0 and (self.state[i * 4 + k] - 1) // 4 == i and (
                                    self.state[i * 4 + k] - 1) % 4 < j:  # 同一行的右边某个点正确位置是在他的正左边
                                h2 += 2
                    if i == row_goal:
                        for k in range(i + 1, 4):
                            if self.state[k * 4 + j] != 0 and (self.state[k * 4 + j] - 1) % 4 == j and (
                                    self.state[k * 4 + j] - 1) // 4 < i:  # 同一列的下边某个点正确位置是在他的正上边
                                h2 += 2

        self.h = (h1 + h2)

    def successors(self):
        """
        Generate the successors of the puzzle.
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
            l = list(self.state)
            l[zero] = l[zero + move[i]]
            l[zero + move[i]] = 0
            puz_child = Puzzle(l)
            puz_child.g = self.g + 1
            puz_child.path = copy.deepcopy(self.path)
            puz_child.path.append(l[zero])
            yield puz_child


def A_star(p):
    """
    A* Algorithm.
    p: the initial state of the puzzle
    sum: the number of nodes searched
    pri: the priority queue
    visited: the set of visited states
    """
    sum = 0
    pri = PriorityQueue()
    pri.put(p)
    visited = set()
    while not pri.empty():
        puz = pri.get()
        sum += 1
        if puz.h == 0:
            print("Finding", sum, "nodes")
            return puz
        visited.add(puz.state)
        for puz_child in puz.successors():
            if puz_child.state not in visited:
                pri.put(puz_child)


def IDA_star(p):
    """
    IDA* Algorithm.
    p: the initial state of the puzzle
    limit: the limit of the search
    path: the state changing of the puzzle
    visited: the set of visited states
    """
    threshold_value = p.h
    states = [p]
    visited = set()
    while True:
        temp_threshold_value = search(states, 0, threshold_value, visited)
        if temp_threshold_value == 'FOUND': return states
        if temp_threshold_value == float('inf'): return []
        threshold_value = temp_threshold_value


def search(states, g, limit, visited):
    """
    Search the puzzle.
    path: the state changing of the puzzle
    g: the cost of the path
    limit: the limit of the search
    visited: the set of visited states
    """
    node = states[-1]
    f = g + node.h
    if f > limit: return f
    if node.h == 0: return 'FOUND'
    min = float('inf')
    for successor in node.successors():
        if successor not in states and successor.state not in visited:
            states.append(successor)
            visited.add(successor.state)
            temp = search(states, g + 1, limit, visited)
            if temp == 'FOUND': return 'FOUND'
            if temp < min: min = temp
            states.pop()
            visited.remove(successor.state)
    return min


def print_puzzle(matrix):
    """
    Print the puzzle.
    matrix: the state of the puzzle
    """
    for i in range(4):
        print(" ".join(str(matrix[4 * i + j]) for j in range(4)))


def move_blank(matrix, move):
    """
    Move the blank.
    matrix: the state of the puzzle
    move: the number to move
    """
    zero_index = matrix.index(0)
    move_index = matrix.index(move)
    matrix[zero_index], matrix[move_index] = matrix[move_index], matrix[zero_index]


def main():
    # matrix = [0, 5, 15, 14, 7, 9, 6, 13, 1, 2, 12, 10, 8, 11, 4, 3]  # example initial state
    with open('input2.txt', 'r') as f:
        lines = f.readlines()
    matrix = [int(num) for line in lines for num in line.split()]
    # print(matrix)
    a = Puzzle(matrix)
    time_start = time.time()
    ans = A_star(a)

    print("A* Algorithm Solution:")
    print(ans.path, end='\nstate:\n')
    for move in ans.path:
        print_puzzle(matrix)
        print("----------")
        move_blank(matrix, move)
    print_puzzle(matrix)
    print("----------")
    # print(ans.path)
    print("Total moves:", ans.g)
    time_end = time.time()
    time_c = time_end - time_start
    print('A*:', 'time cost', time_c, 's')

    time_start = time.time()
    ans = IDA_star(a)

    if ans:
        print("\nIDA* Algorithm Solution:")
        print(ans[-1].path, end='\nstate:\n')
        for node in ans:
            print_puzzle(node.state)
            print("----------")
        print("Total moves:", len(ans) - 1)
    time_end = time.time()
    time_c = time_end - time_start
    print('IDA*:', 'time cost', time_c, 's')


if __name__ == '__main__':
    main()
