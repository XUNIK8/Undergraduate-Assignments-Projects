import networkx as nx

from Draw import draw_result
#计算运行时间
import time
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
        edges.append((point[0],point[1].replace("\n",""),1))#数据中 所有边的权值为1

    #在图中添加边
    G.add_weighted_edges_from(edges)
    #判断连通性
    if not nx.is_connected(G):
        raise("该图为非联通图")

    print("文件导入完成")
    print("图中点数为", G.number_of_nodes())
    print("图中的边数为",G.number_of_edges())

    return G


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
    :返回： 点1、2 合并后的图G
    '''

    # 将 两个合并合后新的点 加入图中 名称继承自两点的名称
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
            if w not in G[node1]:  #非公共相邻点，添加新边 权值不变
                G.add_edge(new_node, w, weight=e["weight"])
            else:   #对于公共相邻点，权值相加
                G[new_node][w]["weight"] += e["weight"]

    G.remove_nodes_from([node1, node2])
    return G

def MinimumCutPhase(G):
    '''
     复现参考文献中的伪代码
    :功能: 求一种st最小割
    :输入: 图G
    :返回:st最小割值:
    '''
    node_list = list(G.nodes())
    PQ={}
    for node in node_list:
        PQ[node]=0
    s=None
    t=None
    while PQ!={}:
        max_key = max(zip(PQ.values(), PQ.keys()))
        u=max_key[1]
        del [PQ[u]]

        s=t
        t=u

        for v,e in G[u].items():
            if v in PQ.keys():
                weight=PQ[v]+e["weight"]
                PQ.update({v:weight})
    cut = 0
    for _, edge in G[t].items():
        cut += edge["weight"]
    return cut,s,t







def Stoer_Wagner(G):
    '''
    :功能: 实现Stoer_Wagner算法求无向无权图的 最小割 及最小割值
    :输入: 图G
    :返回: 图G的全局最小割
    '''
    start = time.time()
    min_cut = 1000000
    phase = 1
    last_node=None
    nodes=list(G.nodes())
    new_G=G.copy()# 复制一份图G用于之后操作（不改变图G本身）


    while True:
        print("在第%i阶段"% phase)
        cut, s, t = MinimumCutPhase(new_G)
        new_G = Merge( new_G, s, t)

        #判断图中仅有两个点时退出循环
        if len(new_G.nodes()) ==2:
            break
        phase += 1
        print("合并后图中点的个数为：%i" % nx.number_of_nodes(new_G))

        if cut < min_cut:
            min_cut = cut
            last_node=t
    partition1=last_node.split("_")
    partition2=list(set(nodes) - set(partition1))
    cut_set= {"子点集1": partition1,"子点集2": partition2}
    print("图的最小割的值为: %i" % min_cut)
    print("图的最小割为")
    print(cut_set)
    end = time.time()

    print("程序运行时间为 ",(end - start),"s")
    return cut_set,cut

#示例文件位置
G=read_txt_to_graph("/Users/zcl271828/Desktop/OR final/data/RodeEU_gcc.txt")



cut_set,_=Stoer_Wagner(G)
'''
以下为画图操作
'''
draw_result(G,cut_set["子点集1"],cut_set["子点集2"])
# #print(G.node())
