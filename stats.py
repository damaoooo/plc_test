import os
import torch
import html
import numpy as np
import networkx as nx
import pygraphviz as pgv
from enum import Enum, unique
import json
import pickle
import logging
import shutil
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

@unique
class NodeType(Enum):
    FunctionCall = 0
    Operator = 1
    Literal = 2
    Param = 3
    Return = 4
    Method = 5
    MethodReturn = 6
    Block = 7
    Local = 8
    Unknown = 9

"""
types = ['LOCAL', 'RETURN', 'IDENTIFIER', 'BLOCK', 'METHOD', 'METHOD_RETURN', 'LITERAL', 'PARAM', 'UNKNOWN',]
ops = ['<operator>.arithmeticShiftRight', '<operator>.logicalShiftLeft', '<operator>.incBy', '<operator>.addressOf', 
        '<operator>.TODO', '<operator>.goto', '<operator>.or', '<operator>.assignment', '<operator>.subtraction', 
        '<operator>.assignmentAnd', '<operator>.assignmentXor', '<operator>.NOP', '<operator>.multiplication', 
        '<operator>.not', '<operator>.compare', '<operator>.logicalShiftRight', '<operator>.negation', '<operator>.division']
"""

class FeatureEncoder:
    def __init__(self, operators: list, types: list) -> None:
        self.op = operators
        self.types = types
        self.total = self.types + self.op
        
    # 不用里面的数据和控制流了，因为ASM里面根本没有这些信息，所以不需要手动构建符号表了
    def encode(self, label: str):
        if label in self.total:
            return self.total.index(label)
        else:
            return len(self.total)


if __name__ == '__main__':

    types = ['LOCAL', 'RETURN', 'IDENTIFIER', 'BLOCK', 'METHOD', 'METHOD_RETURN', 'LITERAL', 'PARAM', 'UNKNOWN',]
    ops = ['<operator>.arithmeticShiftRight', '<operator>.logicalShiftLeft', '<operator>.incBy', '<operator>.addressOf', 
            '<operator>.TODO', '<operator>.goto', '<operator>.or', '<operator>.assignment', '<operator>.subtraction', 
            '<operator>.assignmentAnd', '<operator>.assignmentXor', '<operator>.NOP', '<operator>.multiplication', 
            '<operator>.not', '<operator>.compare', '<operator>.logicalShiftRight', '<operator>.negation', '<operator>.division']

    encoder = FeatureEncoder(operators=ops, types=types)

    adj_len = []

    max_adj_len = 0
    max_feature_len = 0

    root_path = "/home/damaoooo/Project/plc"
    source_path = os.path.join(root_path, "door")
    total_file = []
    total_graph = []
    destination = os.path.join(root_path, "dataset_door")
    
    if os.path.exists(destination):
        shutil.rmtree(destination)
    else:
        os.mkdir(destination)

    for file in os.listdir(source_path):
        opt_level = file[5:].split('-')[-1].split("_")[0]
        arch = file[5:].split('-')[0]
        if os.path.isdir(os.path.join(source_path, file)) and file.startswith("Res0_"):
            for dots in os.listdir(os.path.join(source_path, file)):
                full_path = os.path.join(source_path, file, dots)
                graph = nx.Graph(pgv.AGraph(filename=full_path))

                # 过滤掉无用函数
                if not ("_init_" in graph.name or "_body_" in graph.name):
                    continue

                features = []
                graph = graph.to_undirected()
                edges = graph.edges
                start = [int(x[0]) for x in edges]
                end = [int(x[1]) for x in edges]
                base = min(start + end)
                start = [x - base for x in start]
                end = [x - base for x in end]
                adj = [start, end]

                # 提取特征
                for node in graph.nodes:
                    label: str = html.unescape(graph.nodes[node]['label'])
                    t = label[1:label.find(",")]
                    feature = encoder.encode(t)
                    features.append(feature)
                
                
                result = {"name": graph.name, "adj": adj, "feature": features, "opt": opt_level, "arch": arch}
                adj_len.append(len(adj))

                total_file.append(result)
                s = json.dumps(result)

                max_feature_len = max(max_feature_len, len(encoder.total))
                max_adj_len = max(max_adj_len, len(features))

                with open(os.path.join(destination, f"{graph.name}_{arch}_{opt_level}.json"), "w") as f:
                    f.write(s)
                    f.close()
            
    with open(os.path.join(destination, "!total.pkl"), "wb") as f:
        pickle.dump({"data": total_file, "adj_len": max_adj_len, "feature_len": max_feature_len}, f)
        f.close()
    

    logging.info("All done!")
    logging.info(f"MAX_ADJ: {max_adj_len}, MAX_FEATURE: {max_feature_len}")

