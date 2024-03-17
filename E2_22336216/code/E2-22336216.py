import re
import queue
import copy


def Judge(clause1, clause2):
    # 判断能否变量替换
    hash = []
    for i in range(1, len(clause1)):
        if clause1[i] not in variable and clause2[i] not in variable and clause1[i] == clause2[i]:
            continue
        elif clause1[i] not in variable and clause2[i] in variable:
            hash.append((clause2[i], clause1[i]))
        else:
            return False
    return hash


def MGU(KB, assignment, parent):
    # 归结合一函数
    for i, _ in enumerate(KB):  # 这里用len(KB)不知道为什么就不行
        for j, _ in enumerate(KB):
            if i == j: continue
            for m in range(len(KB[i])):
                for n in range(len(KB[j])):
                    # 谓词相反
                    if ((KB[i][m][0] == '¬' + KB[j][n][0] or KB[j][n][0] == '¬' + KB[i][m][0]) and len(KB[i][m]) == len(
                            KB[j][n])):
                        if KB[i][m][1:] == KB[j][n][1:]:
                            copyclause1 = copy.deepcopy(KB[i])
                            copyclause2 = copy.deepcopy(KB[j])
                            del copyclause1[m]
                            del copyclause2[n]
                            parent.append([i, m, j, n])  # i行第m个 j行第n个
                            assignment.append([])  # 无变量替换
                            newkb = list(map(list, set(map(tuple, (copyclause1 + copyclause2)))))
                            KB.append(newkb)
                            if not newkb:
                                return  # 能归结
                        else:
                            hash = Judge(KB[i][m], KB[j][n])
                            if not hash:
                                continue
                            else:
                                copyclause1 = copy.deepcopy(KB[i])
                                copyclause2 = copy.deepcopy(KB[j])
                                del copyclause1[m]
                                del copyclause2[n]
                                for p in range(len(hash)):
                                    for q in range(len(copyclause2)):
                                        while hash[p][0] in copyclause2[q]:
                                            copyclause2[q][copyclause2[q].index(hash[p][0])] = hash[p][1]

                                parent.append([i, m, j, n])  # i行第m个 j行第n个
                                assignment.append(hash)  # 变量替换
                                newkb = list(map(list, set(map(tuple, (copyclause1 + copyclause2)))))
                                KB.append(newkb)
                                if not newkb:
                                    return  # 能归结


def pruning(n, KB, assignment, parent):
    # 使用二叉树结构层序遍历剪枝
    pruningkb = []
    q = queue.Queue()
    q.put(parent[-1])
    pruningkb.append([KB[-1], parent[-1], assignment[-1]])

    # 只有非知识库内的句子才会有变量替换
    while not q.empty():
        cur = q.get()
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
    # 重新标号，使用字典对应
    newindex = {i: None for i in range(n)}
    for i in range(len(pruningkb)):
        # print(pruningkb[i][1])
        if pruningkb[i][1][0] not in newindex:
            newindex[pruningkb[i][1][0]] = None
        if pruningkb[i][1][2] not in newindex:
            newindex[pruningkb[i][1][2]] = None
    newindex = sorted(newindex.keys())
    newindex = {x: newindex.index(x) + 1 for x in newindex}
    return newindex


def convert_to_string(lst):
    # 变量替换
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
    # KB还原回正常形式
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
    if len(kb[line]) == 1:
        return ''
    else:
        return chr(num + 97)


def stdoutput(n, kb, pruningkb, newindex):
    count = n
    for i, j in enumerate(pruningkb):
        if i == len(pruningkb) - 1:
            print(count + i + 1, f"R[{newindex[j[1][0]]},{newindex[j[1][2]]}] = []")
        else:
            #
            print(count + i + 1,
                  f"R[{newindex[j[1][0]]}{num_to_string(kb, j[1][0], j[1][1])},{newindex[j[1][2]]}{num_to_string(kb, j[1][2], j[1][3])}]{convert_to_string(j[2])} = ",
                  end='')
        for k in range(len(j[0])):
            if k is not len(j[0]) - 1:
                print(restore_string(j[0][k]), end=',')
            else:
                print(restore_string(j[0][k]))


variable = ['x', 'y', 'z', 'u', 'v', 'w']


def main():
    filename = "alpineclub.txt"
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

    # print(KB)

    for i in range(len(KB)):
        for j in range(len(KB[i])):
            KB[i][j] = KB[i][j].replace('(', ",").replace(')', '').split(',')

    # print()
    # for item in KB:
    #     print(item)
    # print()
    MGU(KB, assignment, parent)
    # print()
    # for i, item in enumerate(KB):
    #     print(i,item)
    # print()
    # for item in parent:
    #     print(item)
    # print()

    pruningkb = pruning(n, KB, assignment, parent)

    # for item in pruningkb:
    #     print(item)
    newindex = labeling(n, pruningkb)
    # print(newindex)
    pruningkb = pruningkb[::-1]
    stdoutput(n, KB, pruningkb, newindex)


if __name__ == '__main__':
    main()
