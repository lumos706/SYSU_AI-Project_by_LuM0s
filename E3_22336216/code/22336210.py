import heapq
import time
from queue import PriorityQueue


class Board:
    def __init__(self, blocks, parent=None, moves=0):
        self.blocks = blocks[:]
        self.empty_space = self.blocks.index(0)
        self.parent = parent
        self.moves = moves
        self.moved_number = None

    def moves_parity(self):
        x, y = divmod(self.blocks.index(0), 4)
        return abs(x - 3) + abs(y - 3)

    def move(self, direction):
        x, y = divmod(self.empty_space, 4)
        self.moved_number = None
        if direction == 'up' and x > 0:
            self.blocks[self.empty_space], self.blocks[self.empty_space - 4] = (self.blocks[self.empty_space - 4],
                                                                                self.blocks[self.empty_space])
            self.moved_number = self.blocks[self.empty_space]
            self.empty_space -= 4
            self.moves += 1
        elif direction == 'down' and x < 3:
            self.blocks[self.empty_space], self.blocks[self.empty_space + 4] = (self.blocks[self.empty_space + 4],
                                                                                self.blocks[self.empty_space])
            self.moved_number = self.blocks[self.empty_space]
            self.empty_space += 4
            self.moves += 1
        elif direction == 'left' and y > 0:
            self.blocks[self.empty_space], self.blocks[self.empty_space - 1] = (self.blocks[self.empty_space - 1],
                                                                                self.blocks[self.empty_space])
            self.moved_number = self.blocks[self.empty_space]
            self.empty_space -= 1
            self.moves += 1
        elif direction == 'right' and y < 3:
            self.blocks[self.empty_space], self.blocks[self.empty_space + 1] = (self.blocks[self.empty_space + 1],
                                                                                self.blocks[self.empty_space])
            self.moved_number = self.blocks[self.empty_space]
            self.empty_space += 1
            self.moves += 1

    def h(self):
        distance = 0
        linear_conflict = 0
        for i in range(len(self.blocks)):
            if self.blocks[i] != 0:
                x, y = divmod(i, 4)
                target_x, target_y = divmod(self.blocks[i] - 1, 4)
                distance += abs(x - target_x) + abs(y - target_y)
                # check for linear conflict in rows
                if x == target_x:
                    for j in range(y + 1, 4):
                        if self.blocks[i] > self.blocks[x * 4 + j] > 0 and divmod(self.blocks[x * 4 + j] - 1, 4)[0] == x:
                            linear_conflict += 1
                # check for linear conflict in columns
                if y == target_y:
                    for j in range(x + 1, 4):
                        if self.blocks[i] > self.blocks[j * 4 + y] > 0 and divmod(self.blocks[j * 4 + y] - 1, 4)[1] == y:
                            linear_conflict += 1
        return distance + 2 * linear_conflict

    def g(self):
        return 1.02*self.moves

    def __lt__(self, other):
        return self.h() < other.h()

    def is_goal(self):
        return self.blocks == list(range(1, 16)) + [0]

    def print(self):
        for i in range(4):
            print(' '.join([str(self.blocks[i * 4 + j]) for j in range(4)]))


def a_star(board):
    start = board
    queue = PriorityQueue()
    visited = {str(start.blocks)}
    queue.put((start.h() + start.g(), start))
    while not queue.empty():
        current = queue.get()[1]
        if current.is_goal():
            return current
        for direction in ['up', 'down', 'left', 'right']:
            new_board = Board(current.blocks, current, current.moves)
            new_board.move(direction)
            if new_board.blocks != current.blocks and str(new_board.blocks) not in visited:
                visited.add(str(new_board.blocks))
                queue.put((new_board.h() + new_board.g(), new_board))
    return None


def search(path, g, bound, moves, parity):
    node = path[-1]
    f = g + node.h()
    if f > bound:
        return f, None
    if g % 2 == parity:
        if node.is_goal():
            return 'FOUND', moves
    min_t = float('inf')
    for direction in ['up', 'down', 'left', 'right']:
        new_board = Board(node.blocks, node, node.moves)
        new_board.move(direction)
        if new_board.blocks != node.blocks and (len(path) < 2 or new_board.blocks != path[-2].blocks):
            path.append(new_board)
            t, move = search(path, g + 1, bound, moves + [new_board.moved_number], parity)
            if t == 'FOUND':
                return 'FOUND', move
            if t < min_t:
                min_t = t
            path.pop()
    return min_t, None


def ida_star(root):
    bound = root.h()
    path = [root]
    moves = []
    parity = root.moves_parity() % 2
    while True:
        t, move = search(path, 0, bound, moves, parity)
        if t == 'FOUND':
            return move
        if t == float('inf'):
            return None
        bound = t


def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.read().split('\n')
        temp = [int(num) for line in lines for num in line.split()]
    return Board(temp)


def print_path(board):
    if board.parent is not None:
        print_path(board.parent)
    if board.moved_number is not None:
        print(board.moved_number, end=" ")


start_board = read_file('input2.txt')
start_board.print()
start_time1 = time.time()
result1 = a_star(start_board)
print("A* Solution: ")
print_path(result1)
print()
print("Moves: ", result1.moves)
end_time1 = time.time()
print("Running time of A*: ", end_time1 - start_time1, "seconds")
print()
start_time2 = time.time()
result2 = ida_star(start_board)
print("IDA* Solution: ")
for i in result2:
    print(i, end=" ")
print()
print("Moves: ", len(result2))
end_time2 = time.time()
print("Running time of IDA*: ", end_time2 - start_time2, "seconds")
