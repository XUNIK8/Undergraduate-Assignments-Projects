import networkx as nx
import random
from Draw import draw_process,draw_result


import time
'''


'''
def read_txt_to_graph(path):
    '''

    :功能：读取表示图txt文件，将其转化为NX包的图对象。其中点的表示形式为字符串
         txt文件格式：每行代表一条边,两个数字为边的两个端点
    :输入： 文件路径
    :返回： nx图对象
    '''

    # 创建图
    G=nx.Graph()
    with open(path, "r") as f:
        data = f.readlines()

    edges=[]
    for items in data:
        # 处理字符串
        point=items.split(" ")
        edges.append((point[0],point[1].replace("\n",""),1))

    #在图中添加边
    G.add_weighted_edges_from(edges)
    print("文件导入完成")
    print("图中点数为",G.number_of_nodes())
    print("图中的边数为",G.number_of_edges())

    return G





def Contraction(G):
    '''

    :功能：求图G的一种割
    :输入：图G:
    :返回：:
    '''
    #新建一个图对象进行操作
    new=G.copy()
    phase=1
    while new.number_of_nodes()>2:
        #随机选取一条边
        u,v= random.choice(list(new.edges()))
        #合并这条边的两段点
        new = Merge(new,u,v)

        phase+=1

    cut_value=list(new.edges(data=True))[0][-1]["weight"]
    node_list=list(new.nodes())
    cut_set= {"子点集1":node_list[0].split("_"),"子点集2":node_list[1].split("_")}#得到割集 点的划分（以字典的形式）
    #仅有两个点，这两个点的名称中含有割集的划分
    return cut_value,cut_set


def Merge(G,node1,node2):
    '''
    :功能：合并给定的两点
          具体处理方式如下：
              1.添加新点 代替要合并的两点
              2.将新点与两点的相邻点连接，权值不变 若为公共相邻点，权值相加.
                忽略两待合并点之间的边（保证无Self-loop）
              3.在G中将node1 node2 删除
    :输入1： 图G
    :输入2： 待合并点 node1
    :输入3： 待合并点 node1
    :返回： 点1、2 很冰后的图G
    '''

    # 将 两个点合并后新的点 加入图中 名称继承自两点的名称
    new_node = str(node1) + "_" + str(node2) #字符串相连
    G.add_node(new_node)
    # 在node1的相邻点中遍历
    for w, e in G[node1].items():
        #忽略 node1 与 node2 之间的连接
        if w != node2:
            if w not in G[node2]:      #非公共相邻点，添加新边 权值不变
                G.add_edge(new_node, w, weight=e["weight"])
            else:                      #公共相邻点 ，添加新边
                G.add_edge(new_node, w, weight=e["weight"])
    # 在node2 的相邻点中便利
    for w, e in G[node2].items():
        if w != node1:
            if w not in G[node1]:  #
                G.add_edge(new_node, w, weight=e["weight"])
            else:   #对于公共相邻点，权值相加
                G[new_node][w]["weight"] += e["weight"]

    G.remove_nodes_from([node1, node2])
    return G


def Karger(G):
    '''
    功能：实现Karger算法求无向无权图的 最小割 及最小割值
    :输入 图G:
    :返回 最小割值 及割集:
    '''
    start = time.time()
    random.seed(1)
    iter=100# 迭代次数
    min_cut=10000
    min_cut_set={}
    for i in range(iter):
        print("第%i次迭代" %(i+1))

        cur_cut,cur_cut_set=Contraction(G)
        print("割值为 %i"% cur_cut)
        #print(c)
        if cur_cut< min_cut:
            min_cut=cur_cut
            min_cut_set=cur_cut_set
    end = time.time()
    print("图的最小割为")
    print(min_cut_set)
    print("求得最小割值为：%i" % min_cut)
    print("程序运行时间为 ", (end - start), "s")
    return min_cut_set,min_cut

#示例文件位置
G=read_txt_to_graph("/Users/zcl271828/Desktop/OR final/data/RodeEU_gcc.txt")
cut_set,_=Karger(G)
'''
以下为画图操作
'''
draw_result(G,cut_set["子点集1"],cut_set["子点集2"])