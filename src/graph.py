from __future__ import division
import preprocessing

import networkx as nx

# n=size of window

# create graph for each tweet
#n=window size
def create_graph(nodes, n=3):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    # start from 1. first node doesn't have preceding node =>no edge
    for i in range(1, len(nodes)):
        w_size = n
        # need to draw edge for n consequent nodes
        # if node index is less than n
        # set window size to node index
        if i - n < 0:
            w_size = i
        for p in range(1, w_size + 1):
            node1 = nodes[i - p]
            node2 = nodes[i]
            if g.has_edge(node1, node2):
                # weight proportional to cooccurence rate
                # pair of n-grams occur more than 1 time in the same tweet increment weight by 1
                g[node1][node2]['weight'] += 1
            else:
                g.add_edge(node1, node2, weight=1)
    return g
    # pl.figure(figsize=(10,10))
    # nx.draw(g,with_labels=True, pos=nx.circular_layout(g))
    # pl.show()


# merge newly created graph with main graph
# i=number of tweets in main graph (or ith Tweet)
def merge_graph(tweet_graph, i, class_graph):
    # existing nodes
    class_nodes = class_graph.nodes()
    # tweet nodes
    tweet_nodes = tweet_graph.nodes()
    # add nodes to class graph
    nodes_to_add = [n for n in tweet_nodes if n not in class_nodes]
    class_graph.add_nodes_from(nodes_to_add)

    for u, v, weight in tweet_graph.edges_iter(data=True):
        # check if edge exist in main graph
        if class_graph.has_edge(u, v):
            # calculate weight for edge
            # class_graph weight value for edge
            cl_g_weight = class_graph.get_edge_data(u, v)['weight']
            #print cl_g_weight
            # tweet graph weight value for edge
            tw_g_weight = weight['weight']
            #print tw_g_weight, i
            # weight value
            w = cl_g_weight + (tw_g_weight - cl_g_weight) / i
            class_graph.add_edge(u, v, weight=w)
        else:
            class_graph.add_edge(u, v, weight)

#n=window size
#tokens=tweet tokens, [[tweet_tokens],[tweet_tokens]...]
def build_class_graph(tokens,n=3):
    graph = nx.Graph()
    count = 1
    for i in tokens:
        #print len(i), len(set(i))
        # create graph for tweet
        g = create_graph(i, n)
        merge_graph(g, count, graph)
        count += 1
    return graph


# remove common edges: leave edge in the graph with highest weight
def remove_edge(g, u, v):
    if len(g) > 1:
        for i in g:
            i.remove_edge(u, v)
    else:
        g[0].remove_edge(u, v)


def filter_edges(pos_graph, neg_graph, neut_graph):
    common_edges = set(pos_graph.edges()) & set(neg_graph.edges()) & set(neut_graph.edges())
    for u, v in common_edges:
        pos = pos_graph.get_edge_data(u, v)['weight']
        neg = neg_graph.get_edge_data(u, v)['weight']
        neut = neut_graph.get_edge_data(u, v)['weight']
        if (pos == neg) and (neut == pos):
            continue
        temp = {}
        temp['pos'] = pos
        temp['neg'] = neg
        temp['neut'] = neut
        max_k = max(temp.keys(), key=(lambda k: temp[k]))
        min_k = [k for k in temp.keys() if temp[k] < temp[max_k]]
        graphs = []
        if 'pos' in min_k:
            graphs.append(pos_graph)
        elif 'neg' in min_k:
            graphs.append(neg_graph)
        elif 'neut' in min_k:
            graphs.append(neut_graph)

        remove_edge(graphs, u, v)
    return pos_graph, neg_graph, neut_graph


# Containment Similarity (CS)
def calc_cs(tweet_graph, class_graph):
    tweet_edges = tweet_graph.edges()
    class_edges = class_graph.edges()
    common_edg = list(set(tweet_edges) & set(class_edges))
    cs = len(common_edg) / len(tweet_edges)
    return cs


# Value Similarity (VS)
def calc_vs(tweet_graph, class_graph):
    tweet_edges = tweet_graph.edges()
    class_edges = class_graph.edges()
    common_edg = list(set(tweet_edges) & set(class_edges))
    # calculate value ratio
    vr = 0
    for u, v in common_edg:
        cl_g_weight = class_graph.get_edge_data(u, v)['weight']
        tw_g_weight = tweet_graph.get_edge_data(u, v)['weight']
        vr = vr + min(cl_g_weight, tw_g_weight) / max(cl_g_weight, tw_g_weight)
    vs = vr / len(class_edges)
    return vs


# Normalized Value Similarity (NVS)
def calc_nvs(tweet_graph, class_graph, vs):
    # size similarity
    tweet_edges = tweet_graph.edges()
    class_edges = class_graph.edges()

    ss = len(tweet_edges) / len(class_edges)
    nvs = vs / ss
    return nvs
