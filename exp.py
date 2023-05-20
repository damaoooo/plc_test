import os
import json
import matplotlib.pyplot as plt
from collections import Counter

names = []
good_fun = []
path = "./c_door"

for file in os.listdir(path):
    if file.endswith(".json"):
        
        with open(os.path.join(path, file), 'r') as f:
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