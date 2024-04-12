def Search(board, EMPTY, BLACK, WHITE, isblack):
    # 目前 AI 的行为是随机落子，请实现 AlphaBetaSearch 函数后注释掉现在的 return 
    # 语句，让函数调用你实现的 alpha-beta 剪枝
    # return RandomSearch(board, EMPTY)
    return AlphaBetaSearch(board, EMPTY, BLACK, WHITE, isblack)


def RandomSearch(board, EMPTY):
    # AI 的占位行为，随机选择一个位置落子
    # 在实现 alpha-beta 剪枝中不需要使用
    from random import randint
    ROWS = len(board)
    x = randint(0, ROWS - 1)
    y = randint(0, ROWS - 1)
    while board[x][y] != EMPTY:
        x = randint(0, ROWS - 1)
        y = randint(0, ROWS - 1)
    return x, y, 0


def AlphaBetaSearch(board, EMPTY, BLACK, WHITE, isblack, depth=2):
    def max_value(board, alpha, beta, depth):
        if depth == 0 or is_board_full(board):
            return evaluate(board, isblack), None
        value = float('-inf')
        move = (0, 0)
        for x, y, next_board in get_successors(board, BLACK if isblack else WHITE):
            next_value, _ = min_value(next_board, alpha, beta, depth - 1)
            if next_value > value:
                value, move = next_value, (x, y)
            if value >= beta:
                return value, move
            alpha = max(alpha, value)
        return value, move

    def min_value(board, alpha, beta, depth):
        if depth == 0 or is_board_full(board):
            return evaluate(board, isblack), None
        value = float('inf')
        move = (0, 0)
        for x, y, next_board in get_successors(board, WHITE if isblack else BLACK):
            next_value, _ = max_value(next_board, alpha, beta, depth - 1)
            if next_value < value:
                value, move = next_value, (x, y)
            if value <= alpha:
                return value, move
            beta = min(beta, value)
        return value, move

    alpha, move = max_value(board, float('-inf'), float('inf'), depth)
    return move[0], move[1], alpha


def is_board_full(board, EMPTY=-1):
    return not any(EMPTY in row for row in board)

# 你可能还需要定义评价函数或者别的什么
# =============你的代码=============


def position_value(i, j, ROWS):
    # 计算位置的价值，棋盘中心的价值最高，为7，每离中心远一格，价值减1
    return min(i, ROWS - 1 - i, j, ROWS - 1 - j)


def evaluate(board, isblack):
    # 定义棋型的分数
    SCORES = {
        "FIVE": 999900000,  # 连五
        "FOUR": 33300000,  # 活四
        "SFOUR": 6250000,  # 冲四
        "THREE": 625000,  # 活三
        "STHREE": 12500,  # 眠三
        "TWO": 250,  # 活二
        "STWO": 25,  # 眠二
    }

    # 定义棋型的模式
    PATTERNS = {
        "FIVE": [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
        "FOUR": [[1, 0, 0, 0, 0, -1], [0, 1, 1, 1, 1, -1], [1, 1, -1, 1, 1]],
        "SFOUR": [[0, 0, 0, 0, 0, -1], [0, 1, 1, 1, 1, -1]],
        "THREE": [[-1, 0, 0, 0, -1, -1], [-1, 1, 1, 1, -1, -1]],
        "STHREE": [[-1, 0, 0, 0, -1, 1], [-1, 1, 1, 1, -1, 1]],
        "TWO": [[-1, 0, 0, -1, -1, -1], [-1, 1, 1, -1, -1, -1]],
        "STWO": [[-1, 0, 0, -1, -1, 1], [-1, 1, 1, -1, -1, 1]],
    }

    # 初始化分数
    score = 0

    # 遍历棋盘
    for i in range(len(board)):
        for j in range(len(board[0])):
            # 检查四个方向
            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # 获取线段
                line = [get(board, i + k * dx, j + k * dy) for k in range(-5, 6)]
                # 检查每种棋型
                for pattern, values in PATTERNS.items():
                    for value in values:
                        if (line[5] == value[0] or line[5] == -1):
                            if line[1:6] == value:
                                score += SCORES[pattern] * (1 if isblack else -1)
                        value = [-v for v in value]  # 反转棋型，用于检查另一种颜色的棋子
            # 加上位置价值
            if board[i][j] != -1:
                score += position_value(i, j, len(board)) * (1 if board[i][j] == 1 else -1)
                print(score)
    return score


def get(board, i, j):
    # 获取棋盘上的值，如果超出边界则返回0
    if 0 <= i < len(board) and 0 <= j < len(board[0]):
        return board[i][j]
    return 0

# 以下为编写搜索和评价函数时可能会用到的函数，请看情况使用、修改和优化
# =============辅助函数=============


def _coordinate_priority(coordinate):
    x, y = coordinate[0], coordinate[1]
    center = 15 // 2
    # 计算到中心的欧氏距离
    distance = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
    # 返回负距离作为优先级
    return distance


def get_successors(board, color, priority=_coordinate_priority, EMPTY=-1):
    '''
    返回当前状态的所有后继（默认按坐标顺序从左往右，从上往下）
    ---------------参数---------------
    board       当前的局面，是 15×15 的二维 list，表示棋盘
    color       当前轮到的颜色
    EMPTY       空格在 board 中的表示，默认为 -1
    priority    判断落子坐标优先级的函数（结果为小的优先）
    ---------------返回---------------
    一个生成器，每次迭代返回一个的后继状态 (x, y, next_board)
        x           落子的 x 坐标（行数/第一维）
        y           落子的 y 坐标（列数/第二维）
        next_board  后继棋盘
    '''
    # 注意：生成器返回的所有 next_board 是同一个 list！
    from copy import deepcopy
    next_board = deepcopy(board)
    ROWS = len(board)
    idx_list = [(x, y) for x in range(15) for y in range(15)]
    idx_list.sort(key=priority)
    # print(idx_list)
    for x, y in idx_list:
        if board[x][y] == EMPTY:
            next_board[x][y] = color
            yield (x, y, next_board)
            next_board[x][y] = EMPTY


# 这是使用 successors 函数的一个例子，打印所有后继棋盘
def _test_print_successors():
    '''
    棋盘：
      0 y 1   2
    0 1---+---1
    x |   |   |
    1 +---0---0
      |   |   |
    2 +---+---1
    本步轮到 1 下
    '''
    board = [
        [1, -1, 1],
        [-1, 0, 0],
        [-1, -1, 1]]
    EMPTY = -1
    next_states = get_successors(board, 1)
    for x, y, state in next_states:
        print(x, y, state)
    # 输出：
    # 0 1 [[1, 1, 1], [-1, 0, 0], [-1, -1, 1]]
    # 1 0 [[1, -1, 1], [1, 0, 0], [-1, -1, 1]]
    # 2 0 [[1, -1, 1], [-1, 0, 0], [1, -1, 1]]
    # 2 1 [[1, -1, 1], [-1, 0, 0], [-1, 1, 1]]


def get_next_move_locations(board, EMPTY=-1):
    '''
    获取下一步的所有可能落子位置
    ---------------参数---------------
    board       当前的局面，是 15×15 的二维 list，表示棋盘
    EMPTY       空格在 board 中的表示，默认为 -1
    ---------------返回---------------
    一个由 tuple 组成的 list，每个 tuple 代表一个可下的坐标
    '''
    next_move_locations = []
    ROWS = len(board)
    for x in range(ROWS):
        for y in range(ROWS):
            if board[x][y] != EMPTY:
                next_move_locations.append((x, y))
    return next_move_locations


def get_pattern_locations(board, pattern):
    '''
    获取给定的棋子排列所在的位置
    ---------------参数---------------
    board       当前的局面，是 15×15 的二维 list，表示棋盘
    pattern     代表需要找的排列的 tuple
    ---------------返回---------------
    一个由 tuple 组成的 list，每个 tuple 代表在棋盘中找到的一个棋子排列
        tuple 的第 0 维     棋子排列的初始 x 坐标（行数/第一维）
        tuple 的第 1 维     棋子排列的初始 y 坐标（列数/第二维）
        tuple 的第 2 维     棋子排列的方向，0 为向下，1 为向右，2 为右下，3 为左下；
                            仅对不对称排列：4 为向上，5 为向左，6 为左上，7 为右上；
                            仅对长度为 1 的排列：方向默认为 0
    ---------------示例---------------
    对于以下的 board（W 为白子，B为黑子）
      0 y 1   2   3   4   ...
    0 +---W---+---+---+-- ...
    x |   |   |   |   |   ...
    1 +---+---B---+---+-- ...
      |   |   |   |   |   ...
    2 +---+---+---W---+-- ...
      |   |   |   |   |   ...
    3 +---+---+---+---+-- ...
      |   |   |   |   |   ...
    ...
    和要找的 pattern (WHITE, BLACK, WHITE)：
    函数输出的 list 会包含 (0, 1, 2) 这一元组，代表在 (0, 1) 的向右下方向找到了
    一个对应 pattern 的棋子排列。
    '''
    ROWS = len(board)
    DIRE = [(1, 0), (0, 1), (1, 1), (1, -1)]
    pattern_list = []
    palindrome = True if tuple(reversed(pattern)) == pattern else False
    for x in range(ROWS):
        for y in range(ROWS):
            if pattern[0] == board[x][y]:
                if len(pattern) == 1:
                    pattern_list.append((x, y, 0))
                else:
                    for dire_flag, dire in enumerate(DIRE):
                        if _check_pattern(board, ROWS, x, y, pattern, dire[0], dire[1]):
                            pattern_list.append((x, y, dire_flag))
                    if not palindrome:
                        for dire_flag, dire in enumerate(DIRE):
                            if _check_pattern(board, ROWS, x, y, pattern, -dire[0], -dire[1]):
                                pattern_list.append((x, y, dire_flag + 4))
    return pattern_list


# get_pattern_locations 调用的函数
def _check_pattern(board, ROWS, x, y, pattern, dx, dy):
    for goal in pattern[1:]:
        x, y = x + dx, y + dy
        if x < 0 or y < 0 or x >= ROWS or y >= ROWS or board[x][y] != goal:
            return False
    return True


def count_pattern(board, pattern):
    # 获取给定的棋子排列的个数
    return len(get_pattern_locations(board, pattern))


def is_win(board, color, EMPTY=-1):
    # 检查在当前 board 中 color 是否胜利
    pattern1 = (color, color, color, color, color)  # 检查五子相连
    # pattern2 = (EMPTY, color, color, color, color, EMPTY)  # 检查「活四」
    return count_pattern(board, pattern1) > 0


# 这是使用以上函数的一个例子
def _test_find_pattern():
    '''
    棋盘：
      0 y 1   2   3   4   5
    0 1---+---1---+---+---+
    x |   |   |   |   |   |
    1 +---0---0---0---0---+ ... 此行有 0 的「活四」
      |   |   |   |   |   |
    2 +---+---1---+---+---1
      |   |   |   |   |   |
    3 +---+---+---+---0---+
      |   |   |   |   |   |
    4 +---+---+---1---0---1
      |   |   |   |   |   |
    5 +---+---+---+---+---+
    '''
    board = [
        [1, -1, 1, -1, -1, -1],
        [-1, 0, 0, 0, 0, -1],
        [-1, -1, 1, -1, -1, 1],
        [-1, -1, -1, -1, 0, -1],
        [-1, -1, -1, 1, 0, 1],
        [-1, -1, -1, -1, -1, -1]]
    pattern = (1, 0, 1)
    pattern_list = get_pattern_locations(board, pattern)
    assert pattern_list == [(0, 0, 2), (0, 2, 0), (2, 5, 3), (4, 3, 1)]
    # (0, 0) 处有向右下的 pattern
    # (0, 2) 处有向下方的 pattern
    # (2, 5) 处有向左下的 pattern
    # (4, 3) 处有向右方的 pattern
    assert count_pattern(board, (1,)) == 6
    # 6 个 1
    assert count_pattern(board, (1, 0)) == 13
    # [(0, 0, 2), (0, 2, 0), (0, 2, 2), (0, 2, 3), (2, 2, 4),
    #  (2, 2, 6), (2, 2, 7), (2, 5, 3), (2, 5, 6), (4, 3, 1),
    #  (4, 3, 7), (4, 5, 5), (4, 5, 6)]
    assert is_win(board, 1) == False
    # 1 没有达到胜利条件
    assert is_win(board, 0) == True
    # 0 有「活四」，胜利
