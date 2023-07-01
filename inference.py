import torch
import os
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from model import PLModelForAST
from dataset import ASTGraphDataModule
from collections import Counter
from typing import Dict


def get_cos_similar_multi(v1, v2):
    num = np.dot([v1], v2.T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    res = num / denom
    return 0.5 + 0.5 * res


class FunctionEmbedding:
    def __init__(self, name: str, adj, feature, embedding):
        self.name = name
        self.adj = adj
        self.feature = feature
        self.embedding = embedding
        self.cosine = 0


class ModelConfig:
    def __init__(self):
        # file location
        self.model_path = "lightning_logs/c_re_door/checkpoints/epoch=59-step=42554.ckpt"
        self.dataset_path = "./c_language"
        self.exclude_list = []

        # Runtime Setting
        self.cuda = False

        # If no dataset is provided
        self.feature_length = 139
        self.max_length = 1000

        # Model Hyper-Parameters
        self.alpha = 0.2
        self.dropout = 0.3
        self.hidden_features = 64
        self.n_heads = 6
        self.output_features = 128

        # Predict
        self.topK = 3


class InferenceModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        if self.config.dataset_path:
            total_path = os.path.join(self.config.dataset_path, "!total.pkl")
            self.dataset = ASTGraphDataModule(total_path, exclude=self.config.exclude_list, batch_size=1)
            self.dataset.prepare_data()
            self.feature_length = self.dataset.feature_length
            self.max_length = self.dataset.max_length
        else:
            self.feature_length = self.config.feature_length
            self.max_length = self.config.max_length
            self.dataset = None

        self.model = PLModelForAST(adj_length=self.max_length, in_features=self.feature_length,
                                   hidden_features=self.config.hidden_features,
                                   alpha=self.config.alpha, dropout=self.config.dropout, n_heads=self.config.n_heads,
                                   output_features=self.config.hidden_features
                                   ).load_from_checkpoint(self.config.model_path)
        if self.config.cuda:
            self.model = self.model.cuda()
        self.model.eval()

    def data_distribution(self):
        if not self.dataset:
            print("dataset is not specified")
            return

        names = []
        adj_len = []

        for file in os.listdir(self.config.dataset_path):
            if file.endswith(".json"):
                function_name = file.split("__")[0]
                names.append(function_name)
                with open(os.path.join(self.config.dataset_path, file), 'r') as f:
                    content = f.read()
                    f.close()
                content = json.loads(content)
                fea = content["feature"]
                adj_len.append(len(fea))

        plt.boxplot(adj_len)
        return

    def bad_function_list(self, path):
        names = []
        good_fun = []
        # path = self.config.dataset_path
        if not path:
            print("dataset is not specified")
            return

        for directory in os.listdir(path):
            if not os.path.isdir(os.path.join(path, directory)):
                continue
            for file in os.listdir(os.path.join(path, directory, "c_cpg")):
                if file.endswith(".json"):
                    with open(os.path.join(path, directory, "c_cpg", file), 'r') as f:
                        content = f.read()
                        f.close()
                    content = json.loads(content)
                    function_name = content['name'].split("__")[0]
                    names.append(function_name)
                    adj = content["adj"]
                    if len(adj[0]) < 1000:
                        good_fun.append(function_name)

        print("Exclusive: ", set(names) - set(good_fun))
        print("Count:", Counter(good_fun))

    def get_function_list_embedding(self, file_dir):

        function_set: Dict[str, FunctionEmbedding] = {}

        for file in os.listdir(file_dir):
            if file.endswith('.json'):
                try:
                    name, embedding = self.get_function_embeding(os.path.join(file_dir, file))
                    function_set[name] = embedding
                except IndexError:
                    with open(os.path.join(file_dir, file), 'r') as f:
                        content = f.read()
                        content = json.loads(content)
                        f.close()
                    name = content['name']
                    print("Function {} is too long with length {}".format(name, len(content['feature'])))
                
        return function_set

    @torch.no_grad()
    def get_top_k(self, file_dir, function_filename):

        function_set = self.get_function_list_embedding(file_dir=file_dir)

        name, embedding = self.get_function_embeding(function_filename)

        result = []

        for k in function_set:
            cosine = get_cos_similar_multi(embedding.embedding, function_set[k].embedding)
            result.append([k, cosine])
            function_set[k].cosine = cosine

        result = sorted(result, key=lambda x: x[1])[-self.config.topK:][::-1]
        result = ['Function: {}, Cosine: {}'.format(x[0], round(x[1].item(), 4)) for x in result]
        print(name, 'in', result)

    def get_function_embeding(self, json_file):
        with open(json_file, 'r') as f:
            content = f.read()
            content = json.loads(content)
            f.close()
        feature, adj, name = self.to_tensor(content)
        embedding = self.model.my_model(adj, feature).cpu().numpy()
        return name, FunctionEmbedding(name=name, adj=adj, feature=feature, embedding=embedding)

    def to_tensor(self, json_dict: dict):
        adj = json_dict['adj']
        feature = json_dict['feature']
        name = json_dict['name']

        feature = torch.FloatTensor(feature)
        adj_matrix = torch.zeros([self.max_length, self.max_length])
        start = torch.tensor(adj[0])
        end = torch.tensor(adj[1])
        adj_matrix[start, end] = 1

        adj_matrix = (adj_matrix + adj_matrix.T)
        adj_matrix += torch.eye(self.max_length)

        # For padding
        feature_padder = torch.nn.ZeroPad2d([0, self.feature_length - feature.shape[1], 0, self.max_length - feature.shape[0]])
        feature = feature_padder(feature)[:self.max_length][:, :self.feature_length]

        if self.config.cuda:
            feature = feature.to(device='cuda')
            adj_matrix = adj_matrix.to(device='cuda')

        return feature, adj_matrix, name
    
    @torch.no_grad()
    def get_test_pairs(self, function_list1: str, function_list2: str):

        function_set1 = self.get_function_list_embedding(file_dir=function_list1)
        function_set2 = self.get_function_list_embedding(file_dir=function_list2)

        common_function = list(set(function_set1.keys()).intersection(set(function_set2.keys())))
        exlusive = ['TOF_body', 'PID_body', 'CTUD_ULINT_body', 'RAMP_body', 'CTUD_LINT_body', 'TON_body', 'TP_body', "RAMP_init__", "PID_init__"]
        common_function = [x for x in common_function if x not in exlusive]
        mat2 = []
        for c in function_set2:
            mat2.append(function_set2[c].embedding)
        mat2 = np.vstack(mat2)
        
        result: Dict[str, list] = {}

        for c in common_function:
            query = function_set1[c].embedding
            mm = get_cos_similar_multi(query, mat2)
            rank_list = sorted(zip(mm.reshape(-1), list(function_set2.keys())), key=lambda x: x[0], reverse=True)[:self.config.topK]
            result[c] = rank_list.copy()

        correct = 0
        for c in result:
            if self.judge(c, [x[1] for x in result[c]]):
                correct += 1
                # print(c, result[c], "Right")
            # else:
                # print(c, result[c], "Wrong")

        return correct, len(result.keys())
    
    def judge(self, func: str, candidate: list):
        if func in candidate:
            return True
        else:
            if "CTU" in func or "CTD" in func:
                for x in candidate:
                    if "CTU" in x or "CTD" in x:
                        return True

        return False
        

if __name__ == '__main__':
    model_config = ModelConfig()
    model_config.model_path = "lightning_logs/version_2/checkpoints/last.ckpt"
    model_config.dataset_path = ""
    model_config.feature_length = 141
    model_config.cuda = True
    model_config.topK = 10
    model = InferenceModel(model_config)
    
    recall = []
    
    for k in range(1, 11):
    
        correct_total = 0
        total_total = 0
        
        model.config.topK = k

        
        for arch1 in ['g++', 'arm-linux-gnueabi-g++', 'powerpc-linux-gnu-g++', 'mips-linux-gnu-g++']:
            for opt1 in ["O0", "O1", "O2", "O3"]:
                for arch2 in ['g++', 'arm-linux-gnueabi-g++', 'powerpc-linux-gnu-g++', 'mips-linux-gnu-g++']:
                    for opt2 in ["O0", "O1", "O2", "O3"]:
                        file_name1 = "Res0_{}-{}".format(arch1, opt1)
                        file_name2 = "Res0_{}-{}".format(arch2, opt2)
                        correct, total = model.get_test_pairs(f"dataset/door/{file_name1}/c_cpg", f"dataset/door/{file_name2}/c_cpg")  
                        correct_total += correct
                        total_total += total
        
        recall.append(correct_total / total_total)
        
    print(recall)