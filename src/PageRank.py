#!/usr/bin/python

import numpy as np
import networkx as nx
import csv


with open('../datasets/sent_receive.csv', 'r', encoding='utf-8')as f:
    cs = list(csv.reader(f))
cs = cs[1:]
graph = []
for itm in cs:
    e = itm[:0]
    e.extend(itm[1:])
    graph.append(e)
G = nx.Graph(graph)
M = nx.to_numpy_matrix(G)
N = len(G.nodes())
beta = 0.85
M = beta * M + (1 - beta) * np.ones((N, N)) / N
V = np.matrixlib.mat([1.0 / N for i in range(N)])
V = V.T
for i in range(N):
    out_links = np.sum(M[:, i])
    M[:, i] = M[:, i] / out_links

while True:
    Pr = np.matmul(M, V)
    if np.max(np.abs(V - Pr)) < 1e-8:
        break
    V = Pr

nds = list(G.nodes())
for i in range(N):
    print("%s,%.9lf" % (nds[i], V[i][0]))
