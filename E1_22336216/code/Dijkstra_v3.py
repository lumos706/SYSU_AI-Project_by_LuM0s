import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from queue import PriorityQueue
import sys
import time
import threading



def dijkstra(graph, start, end):
    """
    迪杰斯特拉函数，求最短路径
    参数:字典graph,起点start,终点end
    返回:最短路长度dis[end],路径列表path
    
    """
    dis = {node: float('inf') for node in range(len(graph))}
    dis[start] = 0
    pq = PriorityQueue()
    pq.put((0, start))
    pre = {node: None for node in range(len(graph))}

    while not pq.empty():
        cur_dis, cur_node = pq.get()

        # 同时获取索引和值
        for neighbor, weight in enumerate(graph[cur_node]):# 索引和值
            # 过滤掉没有连接的
            if weight < np.inf:
                distance = cur_dis + weight
                if distance < dis[neighbor]:
                    dis[neighbor] = distance
                    pre[neighbor] = cur_node
                    pq.put((distance, neighbor))

    path = []
    current = end
    while current is not None:
        path.insert(0, current)
        current = pre[current]

    return dis[end], path

# 读取输入数据
while True:
    input_line = input().split()
    m, n = int(input_line[0]), int(input_line[1])

    # 映射节点到整数索引
    node_to_index = {}
    
    # 创建一个m*m的二维数组，初始值无穷大
    graph = np.full((m, m), np.inf) 
    
    # 读取边的信息
    for _ in range(n):
        edge_line = input().split()
        node1, node2, weight = edge_line[0], edge_line[1], int(edge_line[2])
        
        node_to_index.setdefault(node1, len(node_to_index))
        node_to_index.setdefault(node2, len(node_to_index))
        index1, index2 = node_to_index[node1], node_to_index[node2]
        
        graph[index1][index2] = weight
        graph[index2][index1] = weight

    # 获取数组形状，graph.shape[0]:行数，graph.shape[1]:列数
    if graph.shape[0] != m:
        print("图中节点数与输入不相等，请重新输入")
    else:
        break

print()
print(graph)
print()

G = nx.Graph()
for i in range(m):
    for j in range(i+1, m):
        if graph[i][j] < np.inf:
            G.add_edge(i, j, weight=graph[i][j])

# 绘制图形
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=10, font_color='black')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

# 处理循环输入
while True:
    input_line = input().split()
    start, end = input_line[0], input_line[1]
    if start == "-1" and end == "-1":
        break
    index_start, index_end = node_to_index[start], node_to_index[end]
    
    start_time = time.time()
    
    shortest_distance, shortest_path = dijkstra(graph, index_start, index_end)
    
    # 记录结束时间
    end_time = time.time()
    # 计算时间差
    elapsed_time = end_time - start_time
    print(f"程序运行时间：{elapsed_time} 秒\n")

    # 将数字索引转换回字符
    shortest_path_chars = [node for node in node_to_index.keys() if node_to_index[node] in shortest_path]
    print(f"最短路径: {'-'.join(shortest_path_chars)}，长度: {shortest_distance}")
    
