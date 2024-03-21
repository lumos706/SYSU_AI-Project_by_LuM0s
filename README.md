<a name="CDigV"></a>
# 中山大学人工智能实验————by Lumos
<a name="UsEhe"></a>
# E1——最短路径算法
**给定无向图，及图上两个节点，求其最短路径及长度**
<a name="ZijcH"></a>
### • 要求
使用Python实现，至少实现Dijkstra算法
<a name="CX82E"></a>
### • 输入（统一格式，便于之后的验收）
第1行：节点数m 边数n（中间用空格隔开，下同）；<br />第2行到第n+1行是边的信息，每行是：节点1名称 节点2名称 边权；<br />第n+2行开始可接受循环输入，每行是：起始节点名称 目标节点名称。
<a name="aty4R"></a>
### • 输出（格式不限）
最短路径及其长度。
<a name="h3rGG"></a>
### • 样例
| 6 8 |
| --- |
| a b 2 |
| a c 3 |
| b d 5 |
| b e 2 |
| c e 5 |
| d e 1 |
| d z 2 |
| e z 4 |
| a z |

<a name="jjNFY"></a>
### • 图示
![image.png](https://cdn.nlark.com/yuque/0/2024/png/40485409/1710947030347-b1538925-fea9-4a0f-b826-ed247b5ea1e8.png#averageHue=%23fdfdfd&clientId=u38ce6883-1279-4&from=paste&height=287&id=ue45f0ebf&originHeight=604&originWidth=905&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=54530&status=done&style=none&taskId=u5e391974-de49-4d0c-812c-b397d313830&title=&width=430)
<a name="n0D0W"></a>
### 实验1 提示
• 使用s=input()这种格式输入时，每次读取一行。<br />• 回忆： s.split()返回将字符串s按空格切分后得到的字符串数组。<br />• 怎么存储数据？<br />• 是否要存储某个点到其他点的距离表？怎么存？<br />• 是否要存储所有点的距离表？怎么存？<br />• （*进阶： numpy数组？）<br />• 文件输入：使用open()函数打开一个文件； txt文件可以作为一整个字符串读入（是否可以用split处理换行？）



<a name="N6DPa"></a>
## E2——归结推理