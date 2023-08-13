import torch
import os
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from model import PLModelForAST
from dataset import ASTGraphDataModule
from collections import Counter
from typing import Dict, List
from sklearn.metrics import roc_auc_score


def get_cos_similar_multi(v1, v2):
    num = np.dot([v1], v2.T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    res = num / denom
    return 0.5 + 0.5 * res


class FunctionEmbedding:
    def __init__(self, name: str, embedding):
        self.name = name
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
        torch.set_float32_matmul_precision('high')
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
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
                                   )
        self.model = self.model.load_from_checkpoint(self.config.model_path)
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

    def get_function_file_embedding(self, file_dir):

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
    
    def get_function_set_embedding(self, function_list: List[dict]):
        function_set: Dict[str, FunctionEmbedding] = {}
        for figure in function_list:
            name, embedding = self.get_single_function_embedding(figure)
            function_set[name] = embedding
        return function_set
    
    def get_function_pool_embedding(self, function_list: Dict[str, List[dict]]):
        function_result_list: List[FunctionEmbedding] = []
        name_list = []
        for function_name in function_list:
            for function_body in function_list[function_name]:
                name, embedding = self.get_single_function_embedding(function_body)
                function_result_list.append(embedding)
                name_list.append(name)
        return name_list, function_result_list
        
    def get_single_function_embedding(self, dicts: dict):
        with torch.no_grad():
            tfeature, tadj, name = self.to_tensor(dicts)
            tembedding = self.model.my_model(tadj, tfeature).detach().cpu()
            
        embedding = tembedding.clone().numpy()
        del tembedding
        
        # adj = tadj.detach().cpu().clone().numpy()
        del tadj
        
        # feature = tfeature.detach().cpu().clone().numpy()
        del tfeature

        return name, FunctionEmbedding(name=name, embedding=embedding)
        
    @torch.no_grad()
    def get_top_k(self, file_dir, function_filename):

        function_set = self.get_function_file_embedding(file_dir=file_dir)

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
        embedding = self.model.my_model(adj, feature).detach().cpu().numpy()
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
    def get_test_pairs_pool_embedding(self, function_list1: List, function_pool: Dict, use_cache=""):
        
        if use_cache and os.path.exists(use_cache):
            with open(use_cache, 'rb') as f:
                result = pickle.load(f)
                pool_name_list, function_pool = result
                f.close()
        else:
            pool_name_list, function_pool = self.get_function_pool_embedding(function_pool)
            if use_cache:
                with open(use_cache, 'wb') as f:
                    pickle.dump([pool_name_list, function_pool], f)
                    f.close()
                    
        function_set = self.get_function_set_embedding(function_list1)
        
        common_function = list(set(pool_name_list).intersection(set(function_set.keys())))
        
        mat2 = []
        for c in function_pool:
            mat2.append(c.embedding)
        mat2 = np.vstack(mat2)
        
        result: Dict[str, list] = {}
        
        for c in common_function:
            query = function_set[c].embedding
            mm = get_cos_similar_multi(query, mat2)
            rank_list = sorted(zip(mm.reshape(-1), pool_name_list), key=lambda x: x[0], reverse=True)[:self.config.topK]
            result[c] = rank_list.copy()
        
        return result
            
        
    
    @torch.no_grad()
    def get_test_pairs(self, function_list1: List, function_list2: List):

        function_set1 = self.get_function_set_embedding(function_list=function_list1)
        function_set2 = self.get_function_set_embedding(function_list=function_list2)

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
            
        return result
    
    def get_recall_score(self, result: Dict[str, list], k: int = 10):

        correct = 0
        for c in result:
            if self.judge(c, [x[1] for x in result[c][:k]]):
                correct += 1
        return correct, len(result.keys())
    
    def get_function_name_list(self, dataset: Dict):
        # dataset = Binary_Name - {"function1" : [], "function2": []}
        result = []
        for function_name in dataset:
            for function_content in dataset[function_name]:
                result.append(function_content)
        return result
    
    def judge(self, func: str, candidate: list):
        if func in candidate:
            return True
        else:
            if "CTU" in func or "CTD" in func:
                for x in candidate:
                    if "CTU" in x or "CTD" in x:
                        return True
        return False
            
    def test_recall_K_pool(self, dataset:dict, max_k=1000, cache_path=""):
        recall = {x: [] for x in range(1, max_k + 1)}
        self.config.topK = max_k
            
        record_total = {x: [0, 0] for x in range(1, max_k + 1)}

        for binary in dataset['data']:

            function_list1 = self.get_function_name_list(dataset['data'][binary])
            function_pool1 = dataset['data'][binary]
            with torch.no_grad():
                # with open("result.pkl", 'rb') as f:

                result = self.get_test_pairs_pool_embedding(function_list1=function_list1, function_pool=function_pool1, use_cache=cache_path)
                    
                for k in range(1, max_k + 1):
                    correct, total = self.get_recall_score(result, k=k)

                    record_total[k][0] += correct
                    record_total[k][1] += total
                                        
                    recall[k].append(correct / total)
        
        avg_recall = []
        recall_avg = []
        for k in range(1, max_k + 1):
            avg_recall.append(record_total[k][0] / record_total[k][1])
            recall_avg.append(np.mean(recall[k]))
        
        print("avg_recall", avg_recall, '\n', "recall_avg", recall_avg, '\n')
        return recall
    
    def test_recall_K_file(self, dataset:dict, max_k: int = 10):
        recall = []
        for k in range(1, max_k + 1):
            correct_total = 0
            total_total = 0
            
            self.config.topK = k

            for binary_name in dataset['data']:
                for arch1 in ['gcc', 'arm-linux-gnueabi', 'powerpc-linux-gnu', 'mips-linux-gnu']:
                    for opt1 in ["-O0", "-O1", "-O2", "-O3"]:
                        for arch2 in ['gcc', 'arm-linux-gnueabi', 'powerpc-linux-gnu', 'mips-linux-gnu']:
                            for opt2 in ["-O0", "-O1", "-O2", "-O3"]:
                                try:
                                    print(f"Doing {arch1}{opt1} vs {arch2}{opt2}, current {correct_total}/{total_total} = {correct_total/(total_total+1)}")
                                    function_list1 = self.get_function_name_list(dataset['data'][binary_name], arch1, opt1)
                                    function_list2 = self.get_function_name_list(dataset['data'][binary_name], arch2, opt2)
                                    with torch.no_grad():
                                        correct, total = self.get_test_pairs(function_list1=function_list1, function_list2=function_list2) 
                                    correct_total += correct
                                    total_total += total
                                except ValueError:
                                    print("No that Architecture and Opt_level")
                                
            recall.append(correct_total / total_total)
            print(f"recall@{k}: {correct_total / total_total}")
        print(recall)
        return recall
    
    def ROC_pair(self, function_list1, function_list2):
        function_set1 = self.get_function_set_embedding(function_list=function_list1)
        function_set2 = self.get_function_set_embedding(function_list=function_list2)
        function2_name_list = list(function_set2.keys())
        
        function2_embedding = []
        for function2_name in function2_name_list:
            function2_embedding.append(function_set2[function2_name].embedding)
        function2_embedding = np.vstack(function2_embedding)
            
        scores = []
        labels = []
        for function1 in list(function_set1):
            mm = get_cos_similar_multi(function_set1[function1].embedding, function2_embedding)
            scores.append(max(mm.reshape(-1)))
            labels.append(int(function1 in function2_name_list))
        return scores, labels
         
    def AUC(self, dataset: dict):
        arches = ['gcc', 'arm-linux-gnueabi', 'powerpc-linux-gnu', 'mips-linux-gnu']
        opts = ["-O0", "-O1", "-O2", "-O3"]
        
        scores = []
        labels = []
        
        for binary_name in dataset['data']:
            for arch1 in arches:
                for opt1 in opts:
                    for arch2 in arches:
                        for opt2 in opts:
                            try:

                                function_list1 = self.get_function_name_list(dataset['data'][binary_name], arch=arch1, opt=opt1)
                                function_list2 = self.get_function_name_list(dataset['data'][binary_name], arch=arch2, opt=opt2)
                                with torch.no_grad():
                                    score, label = self.ROC_pair(function_list1=function_list1, function_list2=function_list2)
                                    
                                scores.extend(score)
                                labels.extend(label)
                                
                            except ValueError:
                                print("No that Architecture and Opt_level")
        roc_score = roc_auc_score(y_true=labels, y_score=scores)
        print(roc_score)
        return roc_score
    
    def AUC_average(self, dataset: dict):
        arches = ['gcc', 'arm-linux-gnueabi', 'powerpc-linux-gnu', 'mips-linux-gnu']
        opts = ["-O0", "-O1", "-O2", "-O3"]
        
        auc_score = []
        
        for binary_name in dataset['data']:
            for arch1 in arches:
                for opt1 in opts:
                    for arch2 in arches:
                        for opt2 in opts:
                            try:

                                function_list1 = self.get_function_name_list(dataset['data'][binary_name], arch=arch1, opt=opt1)
                                function_list2 = self.get_function_name_list(dataset['data'][binary_name], arch=arch2, opt=opt2)
                                with torch.no_grad():
                                    score, label = self.ROC_pair(function_list1=function_list1, function_list2=function_list2)
                                    
                                auc_score.append(roc_auc_score(y_true=label, y_score=score))
                                
                            except ValueError:
                                print("No that Architecture and Opt_level")

        auc_score = np.mean(auc_score)
        return auc_score
        
if __name__ == '__main__':
    model_config = ModelConfig()
    
    with open("dataset/allstar/test_set.pkl", 'rb') as f:
        dataset = pickle.load(f)
        f.close()
    
    model_config.model_path = "lightning_logs/allstar_1/epoch=0-step=319000.ckpt"
    model_config.dataset_path = ""
    model_config.feature_length = 145
    model_config.max_length = 1500
    model_config.cuda = True
    model_config.topK = 50
    model = InferenceModel(model_config)
    
    # model.AUC_average(dataset)
    res = model.test_recall_K_pool(dataset, max_k=50, cache_path="")
    
    with open("./recall_allstar.pkl", 'wb') as f:
        pickle.dump(res, f)
        f.close()
    
    data = [res[i] for i in [1, 5, 10, 20, 30, 40, 50]]
    label = [str(x) for x in [1, 5, 10, 20, 30, 40, 50]]
    plt.boxplot(data, labels=label)
    plt.savefig("recall_allstar.png")