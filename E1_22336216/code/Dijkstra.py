import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from queue import PriorityQueue
import sys
import time
import threading


# dijkstra函数
def dijkstra(graph, start, end):
    """
    迪杰斯特拉函数，求最短路径
    参数:字典graph,起点start,终点end
    返回:最短路长度dis[end],路径列表path
    
    """
    print(f"Start node: {start}, end_node: {end}\n")
    dis = {node: float('inf') for node in graph} # 字典(unordered_map)，初始化为{所有点：无穷}
    dis[start] = 0
    pq = PriorityQueue() # 优先队列，小顶堆
    pq.put((0, start)) # 压入一个元组(0,start)
    pre = {node: None for node in graph} # 一个字典提供给最短路径回溯，初始化为{所有点：空对象}

    # 关键部分，在算法原理讲完了
    while not pq.empty():
        cur_dis, cur_node = pq.get() 
        for neighbor, weight in graph[cur_node]:
            distance = cur_dis + weight
            if distance < dis[neighbor]:
                dis[neighbor] = distance
                pre[neighbor] = cur_node
                pq.put((distance, neighbor))


    path = []
    cur = end
    
    # 不能用!=,判断None应该使用is/not
    while cur is not None: 
        path.insert(0, cur)
        cur = pre[cur]

    return dis[end], path

# 读取输入数据
while True:
    input_line = input("请输入行数和列数:\n").split()
    m, n = int(input_line[0]), int(input_line[1])

    graph = {}

    # 读取边的信息
    for i in range(n):
        edge_line = input("请输入边"+str(i+1)+":\n").split()
        node1, node2, weight = edge_line[0], edge_line[1], int(edge_line[2])
        graph.setdefault(node1, []).append((node2, weight))
        graph.setdefault(node2, []).append((node1, weight))

    # 检查节点数是否相等
    if len(graph) != m:
        print("错误：图中的节点数量与指定的节点数量不匹配，重新输入。")
    else:
        break




# 转换为networkx图
G = nx.Graph()
for node, neighbors in graph.items():
    for neighbor, weight in neighbors:
        G.add_edge(node, neighbor, weight=weight)

# 绘制图形
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=10, font_color='black')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show(block=True)


# 处理循环输入
while True:
    input_line = input("请输入要查询的两个点:\n").split()
    start, end = input_line[0], input_line[1]
    if start == '-1' and end == '-1':
        break
    
    start_time = time.time()
    
    shortest_dis, shortest_path = dijkstra(graph, start, end)
    
    # 记录结束时间
    end_time = time.time()
    # 计算时间差
    elapsed_time = end_time - start_time
    print(f"程序运行时间：{elapsed_time} 秒\n")
    
    print(f"最短路径：{' - '.join(shortest_path)}, 长度：{shortest_dis}")
    




