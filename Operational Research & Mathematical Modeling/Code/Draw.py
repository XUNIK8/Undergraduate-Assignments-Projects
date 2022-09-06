import networkx as nx

import matplotlib.pyplot as plt
import numpy as np


def draw_result(G,partation1,partation2):
    '''
    绘制最终结果
    '''
    plt.figure(dpi=200)
    label={}
    for i in partation1+partation2:
        label[i] = i
    pos = nx.spring_layout(G)
    #nx.draw_networkx_labels(G, pos, label)
    nx.draw_networkx_nodes(G, pos, nodelist=partation1, node_color='limegreen',node_size=20)
    nx.draw_networkx_nodes(G, pos, nodelist=partation2, node_color='gold',node_size=10)
    nx.draw_networkx_edges(G,pos)
    cut_edge_list=[]
    rest_edge=[]
    for edge in G.edges():
        judge1=(edge[0] in partation1) and (edge[1] in partation2)
        judge2=(edge[1] in partation1) and (edge[0] in partation2)
        if judge1 or judge2:
            cut_edge_list.append(edge)
        else:
            rest_edge.append(edge)


    nx.draw_networkx_edges(
        G, pos, edgelist=cut_edge_list, width=4, alpha=0.5, edge_color="b", style="dashed",label={("1","2"):1}
    )
    nx.draw_networkx_edges(G,pos,edgelist=rest_edge,width=0.1,alpha=0.4)
    #nx.draw(G, pos,with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    #print(labels)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("MinimumCut Result")

    plt.show()

def draw_process(G,i):
    '''

    绘制第一个例子中间过程
    '''

    plt.subplot(2, 4, i)
    label = {}
    pos = nx.spring_layout(G)
    for j in G.nodes():
        label[j] = j
    nx.draw_networkx_labels(G, pos, label)
    nx.draw_networkx_nodes(G, pos,node_size=300)
    nx.draw_networkx_edges(G, pos)


    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Phase "+str(i),fontdict={'weight':'normal','size': 25})


