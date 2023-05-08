import os
import urllib.parse
import html
import networkx as nx
import pygraphviz as pgv

pwd = '/home/damaoooo/Project/plc/Res0_g++-O1_re_dot'

res = []

files = os.listdir(pwd)
for file in files:
    graph = nx.Graph(pgv.AGraph(filename=os.path.join(pwd, file)))
    if 'body' in graph.name or 'init' in graph.name:
        res.append(graph.name)
        for node in graph.nodes:
            print(html.unescape(graph.nodes[node]['label']))
        exit(0)

print(res, len(res))
