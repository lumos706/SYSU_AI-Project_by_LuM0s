import re
from functools import partial
import numpy as np

# import time


def Search(board, EMPTY, BLACK, WHITE, isblack):
    # 目前 AI 的行为是随机落子，请实现 AlphaBetaSearch 函数后注释掉现在的 return
    # 语句，让函数调用你实现的 alpha-beta 剪枝
    return AlphaBetaSearch(board, EMPTY, BLACK, WHITE, isblack)


def AlphaBetaSearch(board, EMPTY, BLACK, WHITE, isblack, depth=3):
    # 这个函数使用 alpha-beta 剪枝的方法搜索棋盘，返回最佳的落子位置
    # board 是当前棋盘的状态，EMPTY, BLACK, WHITE 分别代表空格、黑子和白子
    # isblack 是一个布尔值，如果为 True 则代表当前轮到黑子，否则轮到白子
    # depth 是搜索的深度

    # 定义 max_value 和 min_value 函数，分别用于在搜索树中的 MAX 节点和 MIN 节点进行搜索
    # 定义 counts 字典，用于记录搜索的节点数和剪枝的次数
    # 最后返回最佳的落子位置和对应的 alpha 值
    def max_value(board, alpha, beta, depth, counts):
        pattw1 = (0, 0, 0, 0, 0)
        pattw2 = (-1, 0, 0, 0, 0, -1)
        if depth == 0 or count_pattern(board, pattw1) + count_pattern(board, pattw2) > 0:
            return evaluate(board, isblack), None  # 到达叶节点，返回评估值
        value = float('-inf')
        move = None
        priority_func = partial(_coordinate_priority, board=board)  # 优先级函数
        for x, y, next_board in get_successors(board, BLACK if isblack else WHITE, priority=priority_func):
            counts['searchcount'] += 1
            next_value, _ = min_value(next_board, alpha, beta, depth - 1, counts)
            if next_value > value:  # 更新最大值
                value, move = next_value, (x, y)
            if value >= beta:  # 剪枝(alpha 剪枝)
                counts['pruncount'] += 1
                # print_board(next_board)
                # print(f"score: {value}")
                # print(f"{counts['pruncount']}pruning: alpha: {value}, beta: {beta}")
                return value, move
            # if value > alpha:
            #     print_board(next_board)
            #     print(f"Update alpha: {value}")
            alpha = max(alpha, value)
            # print(f"alpha: {alpha}, beta: {beta}")
        return value, move

    def min_value(board, alpha, beta, depth, counts):
        pattb1 = (1, 1, 1, 1, 1)
        pattb2 = (-1, 1, 1, 1, 1, -1)
        if depth == 0 or count_pattern(board, pattb1) > 0 or (
                count_pattern(board, pattb2) > 0 and not is_win(board, WHITE, EMPTY)):
            return evaluate(board, isblack), None  # 到达叶节点，返回评估值
        value = float('inf')
        move = None
        priority_func = partial(_coordinate_priority, board=board)  # 优先级函数
        for x, y, next_board in get_successors(board, WHITE if isblack else BLACK, priority=priority_func):
            counts['searchcount'] += 1
            next_value, _ = max_value(next_board, alpha, beta, depth - 1, counts)
            if next_value < value:  # 更新最小值
                value, move = next_value, (x, y)
            if value <= alpha:  # 剪枝(beta 剪枝)
                counts['pruncount'] += 1
                # print_board(next_board)
                # print(f"score: {value}")
                # print(f"{counts['pruncount']}pruning: alpha: {alpha}, beta: {value}")
                return value, move
            # if value < beta:
            #     print_board(next_board)
            #     print(f"Update beta: {value}")
            beta = min(beta, value)
            # print(f"alpha: {alpha}, beta: {beta}")
        return value, move

    counts = {'pruncount': 0, 'searchcount': 0}
    alpha, move = max_value(board, float('-inf'), float('inf'), depth, counts)
    print(f"搜索次数: {counts['searchcount']}, 剪枝次数: {counts['pruncount']}")
    return move[0], move[1], alpha


# 你可能还需要定义评价函数或者别的什么
# =============你的代码=============

# 以下为评价函数的代码，请不要修改
position_value = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
    [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0],
    [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0],
    [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0],
    [0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0],
    [0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0],
    [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0],
    [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0],
    [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0],
    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

SCORES = {
    "FIVE": 999999999,  # 连五
    "FOUR": 66660000,  # 活四
    "SFOUR": 6660000,  # 冲四
    "THREE": 625000,  # 活三
    "STHREE": 62500,  # 眠三

}

# 定义棋型的模式
PATTERNS = {
    "FIVE": ["11111",
             ],  # 连五
    "FOUR": [
        "211112",  # 活四
    ],
    "SFOUR": [
        "011112",  # 冲四
        "211110",  # 冲四
        "011121",  # 冲四
        "121110",  # 冲四
        "11211",  # 冲四

    ],
    "THREE": [
        "21112",  # 活三
        "211212",  # 活三
    ],
    "STHREE": [
        "01112",  # 眠三
        "21110",  # 眠三
        "011212",  # 眠三
        "212110",  # 眠三
        "012112",  # 眠三
        "211210",  # 眠三
        "11221",  # 眠三
        "12211",  # 眠三
        "12121",  # 眠三
    ],

}


def print_board(board):
    for row in board:
        for cell in row:
            if cell == -1:
                print('.', end=' ')
            elif cell == 1:
                print('B', end=' ')
            elif cell == 0:
                print('W', end=' ')
        print()


def evaluate(board, isblack):
    # 这个函数用于评估棋盘的状态，返回一个分数
    # board 是当前棋盘的状态，isblack 是一个布尔值，如果为 True 则代表当前轮到黑子，否则轮到白子
    # 定义棋型的分数
    # 初始化分数
    score = 0

    # 遍历棋盘
    for i in range(len(board)):
        for j in range(len(board[0])):
            # 检查四个方向
            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                if board[i][j] != -1:
                    # 获取线段
                    line = get_line(board, i, j, dx, dy)
                    # 检查每种棋型
                    for pattern, values in PATTERNS.items():
                        for value in values:
                            # 将line和value都转换为字符串
                            # 使用re库的search函数来查找value在line中的位置
                            if re.search(value, line):
                                score += SCORES[pattern] * 1
                                # if str_value == '211112':
                                #     score += 90000000
                            else:
                                score += 8  # 如果没有匹配到，增加8分
                            # 反转棋型，用于检查另一种颜色的棋子
                            value = value.replace("1", "t").replace("0", "1").replace("t", "0")
                            if re.search(value, line):
                                score += SCORES[pattern] * -1
                            else:
                                score -= 8  # 如果没有匹配到，增加8分
                    # 加上位置价值

                    score += position_value[i][j] * (1 if board[i][j] == 1 else -1)
    # print_board(board)
    # print(f"当前局面评分: {score}")
    return score


def get_line(board, i, j, dx, dy, length=11):
    # 这个函数用于获取棋盘上的一条线段
    # board 是当前棋盘的状态，i 和 j 是线段的起始位置，dx 和 dy 是线段的方向，length 是线段的长度
    line = ''
    for k in range(-5, 6):
        if 0 <= i + k * dx < len(board) and 0 <= j + k * dy < len(board[0]) and board[i + k * dx][j + k * dy] != -1:
            line += str(board[i + k * dx][j + k * dy])
        else:
            line += '2'
    return line


# 以下为编写搜索和评价函数时可能会用到的函数，请看情况使用、修改和优化
# =============辅助函数=============


def _coordinate_priority(coordinate, board, EMPTY=-1):
    x, y = coordinate[0], coordinate[1]
    ROWS = 15
    occupied_coordinates = np.array([(i, j) for i in range(ROWS) for j in range(ROWS) if board[i][j] != EMPTY])
    if len(occupied_coordinates) == 0:
        return ((x - 7) ** 2 + (y - 7) ** 2) ** 0.5
    else:
        distances = np.sqrt(np.sum((occupied_coordinates - np.array([x, y])) ** 2, axis=1))
        min_distance = np.min(distances)
        return min_distance


def get_successors(board, color, priority, EMPTY=-1):
    # 这个函数用于获取当前状态的所有后继状态
    # board 是当前棋盘的状态，color 是当前轮到的颜色，priority 是一个函数，用于确定落子位置的优先级，EMPTY 代表空格
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
    idx_list = [(x, y) for x in range(15) for y in range(15) if board[x][y] == EMPTY]
    idx_list.sort(key=priority)
    # idx_list = idx_list[:49]
    idx_list = [idx for idx in idx_list if 0 < priority(idx) <= 2]
    # print(idx_list)
    for x, y in idx_list:
        next_board[x][y] = color
        yield (x, y, next_board)
        next_board[x][y] = EMPTY


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
    pattern1 = (color, color, color, color, color)
    pattern2 = (EMPTY, color, color, color, color)
    pattern3 = (color, color, color, EMPTY, color)
    pattern4 = (color, color, EMPTY, color, color)
    return count_pattern(board, pattern1) + count_pattern(board, pattern2) + count_pattern(board,
                                                                                           pattern3) + count_pattern(
        board, pattern4) > 0
