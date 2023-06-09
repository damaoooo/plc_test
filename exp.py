import os
import json
import matplotlib.pyplot as plt
from collections import Counter

names = []
good_fun = []
path = "/home/damaoooo/Project/common"
for root in os.listdir(path):
    if not os.path.isdir(os.path.join(path, root)):
        continue
    for file in os.listdir(os.path.join(path, root, "c_cpg")):
        if file.endswith(".json"):
            
            with open(os.path.join(path, root, "c_cpg", file), 'r') as f:
                content = f.read()
                f.close()
            content = json.loads(content)
            function_name = content['name'].split("__")[0]
            names.append(function_name)
            adj = content["adj"]
            if len(adj[0]) < 1000:
                good_fun.append(function_name)

print(list(set(names)), len(list(set(names))))
print(list(set(good_fun)), len(list(set(good_fun))))
print(set(names) - set(good_fun))
print(Counter(good_fun))