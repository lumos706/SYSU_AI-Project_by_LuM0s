import re
import queue
import copy
import time


def Judge(clause1, clause2):
    """
    判断两个子句是否可以进行变量替换。

    参数:
        clause1 (list): 第一个原子公式。
        clause2 (list): 第二个原子公式。

    返回:
        hash (list): 两个公式中对应变量的替换集。
        False (bool): 如果两个原子公式不能进行变量替换。
    """
    hash = []
    for i in range(1, len(clause1)):
        if clause1[i] not in variable and clause2[i] not in variable and clause1[i] == clause2[i]:
            continue
        elif clause1[i] not in variable and clause2[i] in variable:
            hash.append((clause2[i], clause1[i]))
        else:
            return False
    return hash


def merge(KB, parent, assignment, clause1_index, i, clause2_index, j, hash_result=None):
    """
    合并知识库中的两个子句。

    参数:
        KB (list): 知识库。
        parent (list): 知识库中每个子句的父子句列表。
        assignment (list): 知识库中每个子句的变量分配列表。
        clause1_index (int): 知识库中第一个子句的索引。
        i (int): 从第一个子句中删除的原子公式的索引。
        clause2_index (int): 知识库中第二个子句的索引。
        j (int): 从第二个子句中删除的原子公式的索引。
        hash_result (list, optional): 两个子句中对应变量的替换列表。

    返回:
        True (bool): 如果新子句为空。
        False (bool): 如果新子句不为空。
    """
    copyclause1 = copy.deepcopy(KB[clause1_index])
    copyclause2 = copy.deepcopy(KB[clause2_index])
    del copyclause1[i]
    del copyclause2[j]
    if hash_result:
        for hash_pair in hash_result:
            for index, predicate in enumerate(copyclause2):
                while hash_pair[0] in predicate:
                    copyclause2[index][predicate.index(hash_pair[0])] = hash_pair[1]

    parent.append([clause1_index, i, clause2_index, j])
    assignment.append(hash_result if hash_result else [])
    newkb = list(map(list, set(map(tuple, (copyclause1 + copyclause2)))))
    KB.append(newkb)
    if not newkb:
        return True
        # 能归结
    return False


def resolve(KB, assignment, parent):
    """
    对知识库中的子句进行归结。

    参数:
        KB (list): 知识库。
        assignment (list): 知识库中每个子句的变量分配列表。
        parent (list): 知识库中每个子句的父子句列表。
    """
    start_time = time.time()
    for clause1_index, clause1 in enumerate(KB):
        for clause2_index, clause2 in enumerate(KB):
            if clause1_index == clause2_index:
                continue
            for i, predicate1 in enumerate(clause1):
                for j, predicate2 in enumerate(clause2):
                    # 谓词相反
                    if (predicate1[0] == '¬' + predicate2[0] or predicate2[0] == '¬' + predicate1[0]) and len(predicate1) == len(predicate2):
                        if predicate1[1:] == predicate2[1:]:
                            if merge(KB, parent, assignment, clause1_index, i, clause2_index, j):
                                end_time = time.time()  # 记录结束时间
                                elapsed_time = end_time - start_time  # 计算时间差
                                if elapsed_time > 10:  # 如果时间超过10秒
                                    print("无法归结")
                                    exit()  # 终止整个程序
                                return
                        else:
                            hash_result = Judge(predicate1, predicate2)
                            if hash_result:
                                if merge(KB, parent, assignment, clause1_index, i, clause2_index, j, hash_result):
                                    end_time = time.time()  # 记录结束时间
                                    elapsed_time = end_time - start_time  # 计算时间差
                                    if elapsed_time > 10:  # 如果时间超过10秒
                                        print("无法归结")
                                        exit()  # 终止整个程序
                                    return


def pruning(n, KB, assignment, parent):
    """
    对归结树进行剪枝。

    参数:
        n (int): 原始知识库中的子句数量。
        KB (list): 知识库。
        assignment (list): 知识库中每个子句的变量分配列表。
        parent (list): 知识库中每个子句的父子句列表。

    返回:
        pruningkb (list): 剪枝后的知识库。
    """
    pruningkb = []
    q = queue.Queue()
    q.put(parent[-1])
    pruningkb.append([KB[-1], parent[-1], assignment[-1]])

    # 只有非知识库内的句子才会有变量替换
    while not q.empty():
        cur = q.get()
        # 大的先进队列(后推出来)，符合推理常理
        if cur[0] > cur[2]:
            if cur[0] >= n:
                pruningkb.append([KB[cur[0]], parent[cur[0]], assignment[cur[0]]])
                q.put(parent[cur[0]])
            if cur[2] >= n:
                pruningkb.append([KB[cur[2]], parent[cur[2]], assignment[cur[2]]])
                q.put(parent[cur[2]])
        else:
            if cur[2] >= n:
                pruningkb.append([KB[cur[2]], parent[cur[2]], assignment[cur[2]]])
                q.put(parent[cur[2]])
            if cur[0] >= n:
                pruningkb.append([KB[cur[0]], parent[cur[0]], assignment[cur[0]]])
                q.put(parent[cur[0]])
    return pruningkb


def labeling(n, pruningkb):
    """
    对剪枝后的知识库中的子句进行标号。

    参数:
        n (int): 原始知识库中的子句数量。
        pruningkb (list): 剪枝后的知识库。

    返回:
        newindex (dict): 一个字典，将旧的子句索引映射到新的子句索引。
    """
    newindex = {i: None for i in range(n)}
    seen_indexes = set()
    for item in pruningkb:
        indexes = item[1]
        # parent[0] and parent[2]
        for index in (indexes[0], indexes[2]):
            if index not in newindex and index not in seen_indexes:
                newindex[index] = None
                seen_indexes.add(index)
    newindex = sorted(newindex.keys())
    newindex = {x: newindex.index(x) + 1 for x in newindex}
    return newindex


def convert_to_string(lst):
    """
    将变量分配列表转换为字符串。

    参数:
        lst (list): 变量分配列表。

    返回:
        str: 变量分配的字符串表示。
    """
    # 初始化一个空字符串
    result = ""
    # 遍历列表中的元组
    for item in lst:
        # 将元组的第一个元素作为键，第二个元素作为值，拼接成字符串
        result += f"{item[0]}={item[1]},"
    # 去除最后一个逗号
    result = result.rstrip(",")
    # 返回结果字符串
    if result == "":
        return result
    return '(' + result + ')'


def restore_string(lst):
    """
    将谓词列表恢复为字符串。

    参数:
        lst (list): 谓词列表。

    返回:
        str: 谓词的字符串表示。
    """
    # 初始化一个空字符串
    result = " "
    # 遍历列表中的元素
    for i, item in enumerate(lst):
        # 如果是第一个元素，添加开头的字符串
        if i == 0:
            result += item + '('
        else:
            result += item + ','
    # 返回结果字符串
    return result[:-1] + ') '


def num_to_string(kb, line, num):
    """
    将数字转换为字符串。

    参数:
        kb (list): 知识库。
        line (int): 知识库中的子句索引。
        num (int): 要转换的数字。

    返回:
        str: 数字的字符串表示。
    """
    if len(kb[line]) == 1:
        return ''
    else:
        return chr(num + 97)


def stdoutput(n, kb, pruningkb, newindex):
    """
    生成标准输出。

    参数:
        n (int): 原始知识库中的子句数量。
        KB (list): 知识库。
        pruningkb (list): 剪枝后的知识库。
        newindex (dict): 一个字典，将旧的子句索引映射到新的子句索引。

    返回:
        output (list): 标准输出。
    """
    count = n
    for i, j in enumerate(pruningkb):
        if i == len(pruningkb) - 1:
            print(count + i + 1, f"R[{newindex[j[1][0]]},{newindex[j[1][2]]}] = []")
        else:
            #
            print(count + i + 1,
                  f"R[{newindex[j[1][0]]}{num_to_string(kb, j[1][0], j[1][1])},{newindex[j[1][2]]}{num_to_string(kb, j[1][2], j[1][3])}]{convert_to_string(j[2])} =",
                  end='')
        for k in range(len(j[0])):
            if k is not len(j[0]) - 1:
                print(restore_string(j[0][k]), end=',')
            else:
                print(restore_string(j[0][k]))


variable = ['x', 'y', 'z', 'u', 'v', 'w', 'xx', 'yy', 'zz']


def main():
    """
    程序的主函数。
    """
    filename = "blockworld.txt"
    KB = []
    n = 0
    # 打开文件
    with open(filename, 'r', encoding="utf-8") as file:
        # 打印知识库
        # 使用计数器跳过第一行
        line_count = 0
        for i, line in enumerate(file):
            # 如果是第一行，则跳过
            if line_count == 0:
                n = int(line)
                # 获取个数n
                print(n)
                line_count += 1
                continue
            print(i, line.strip())

            # 使用正则表达式匹配谓词及其参数
            matches = (re.findall(r'¬?\w+\(\w+,*\w*\)', line))
            '''
            https://docs.python.org/zh-cn/3/library/re.html
            ¬?：匹配零个或一个否定符号（¬）。?表示前面的元素可选。
            \w+：匹配一个或多个字母、数字或下划线，表示谓词或函数名称。
            \(：匹配左括号。
            \w+：再次匹配一个或多个字母、数字或下划线，表示参数中的第一个元素。
            ,*：匹配零个或多个逗号。
            \w*：匹配零个或多个字母、数字或下划线，表示参数中的其余元素。
            \)：匹配右括号。
            '''
            # 将匹配结果添加到 KB 列表中
            KB.append(matches)
    # 记忆变量替换的列表assignment和记录父子句的列表Parent
    assignment = [[] for _ in range(n)]
    parent = [[] for _ in range(n)]
    sorted(KB)

    for i in range(len(KB)):
        for j in range(len(KB[i])):
            KB[i][j] = KB[i][j].replace('(', ",").replace(')', '').split(',')

    resolve(KB, assignment, parent)
    pruningkb = pruning(n, KB, assignment, parent)
    newindex = labeling(n, pruningkb)
    pruningkb = pruningkb[::-1]
    stdoutput(n, KB, pruningkb, newindex)


if __name__ == '__main__':
    main()
