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
    print(f"Start node: {start}, end_node: {end}\n")
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = PriorityQueue()
    pq.put((0, start))
    pre = {node: None for node in graph}

    while not pq.empty():
        cur_dis, cur_node = pq.get()
        for neighbor, weight in graph[cur_node]:
            distance = cur_dis + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                pre[neighbor] = cur_node
                pq.put((distance, neighbor))

    path = []
    cur = end
    while cur is not None:
        path.insert(0, cur)
        cur = pre[cur]

    return distances[end], path

# 读取文本内容,字符串file_content
with open('text.txt', 'r') as file:
    file_content = file.read()

# 以/n分割file_content,列表input_lines
input_lines = file_content.split('\n')

m, n = int(input_lines[0].split()[0]), int(input_lines[0].split()[1])

graph = {}

if len(set(node for edge_line in input_lines[1:n+1] for node in edge_line.split()[:2])) != m:
    print("错误：图中的节点数量与指定的节点数量不匹配。")
    sys.exit()

# 从第二行开始是点1,点2,边权
graph = {}
for i in range(1, n + 1):
    edge_line = input_lines[i].split()
    node1, node2, weight = edge_line[0], edge_line[1], int(edge_line[2])
    graph.setdefault(node1, []).append((node2, weight))
    graph.setdefault(node2, []).append((node1, weight))
    
    
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
# 从第n行开始是查询
for line in input_lines[n + 1:]:
    if not line:
        continue  # 如果是空的就跳过
    start, end = line.split()
    shortest_dis, shortest_path = dijkstra(graph, start, end)
    print(f"最短路径: {' - '.join(shortest_path)}，长度: {shortest_dis}")