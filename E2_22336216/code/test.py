

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


def resolve(KB, parent, assignment, clause1_index, i, clause2_index, j, hash_result=None):
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


def MGU(KB, assignment, parent):
    # 归结合一函数
    for clause1_index, clause1 in enumerate(KB):
        for clause2_index, clause2 in enumerate(KB):
            if clause1_index == clause2_index:
                continue
            for i, predicate1 in enumerate(clause1):
                for j, predicate2 in enumerate(clause2):
                    # 谓词相反
                    if (predicate1[0] == '¬' + predicate2[0] or predicate2[0] == '¬' + predicate1[0]) and len(predicate1) == len(predicate2):
                        if predicate1[1:] == predicate2[1:]:
                            if resolve(KB, parent, assignment, clause1_index, i, clause2_index, j):
                                return
                        else:
                            hash_result = Judge(predicate1, predicate2)
                            if hash_result:
                                if resolve(KB, parent, assignment, clause1_index, i, clause2_index, j, hash_result):
                                    return


def pruning(n, KB, assignment, parent):
    # 使用二叉树结构层序遍历剪枝
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
    # 重新标号，使用字典对应
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
    input_str = "KB = {(A(tony),),(A(mike),),(A(john),),(L(tony,rain),),(L(tony,snow),),(¬A(x),S(x),C(x)),(¬C(y),¬L(y,rain)),(L(z,snow),¬S(z)),(¬L(tony,u),¬L(mike,u)),(L(tony,v),L(mike,v)),(¬A(w),¬C(w),S(w))}"

    # 去除外部大括号
    input_str = input_str[5:].strip("{}")
    # 按逗号分割每个元组，并处理每个元组
    result = []
    matches = input_str.split('),(')
    for match in matches:
        match = match.strip('(').strip(',')
        result.append(match)

    KB = []
    for i,j in enumerate(result):
        print(i+1, j)
    for element in result:
        matches = (re.findall(r'¬?\w+\(\w+,*\w*\)', element))
        KB.append(matches)

    for i in range(len(KB)):
        for j in range(len(KB[i])):
            KB[i][j] = KB[i][j].replace('(', ",").replace(')', '').split(',')

    # print()
    # for item in KB:
    #     print(item)
    # print()
    n = len(KB)
    # print(n)
    # 记忆变量替换的列表assignment和记录父子句的列表Parent
    assignment = [[] for _ in range(n)]
    parent = [[] for _ in range(n)]
    # print(parent)
    # print(KB)

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
