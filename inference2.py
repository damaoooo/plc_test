import torch
import os
import numpy as np
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import PLModelForAST, pearson_score
from dataset import ASTGraphDataModule
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable
from sklearn.metrics import roc_auc_score
from line_profiler import LineProfiler
# import torch.multiprocessing as multiprocessing
import multiprocessing
import queue
from multiprocessing.pool import ThreadPool
import threading

import dgl

def get_cos_similar_multi(v1, v2):
    num = np.dot([v1], v2.T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    res = num / denom
    return 0.5 + 0.5 * res

def similarity_score(query, vectors):
    distance =  np.linalg.norm(query - vectors, axis=-1)
    score = 1 / (1 + distance)
    return score

def get_pearson_score(query, vectors):
    cov = (query * vectors).mean(axis=-1)
    pearson = cov / (query.std(axis=-1) * vectors.std(axis=-1))
    return abs(pearson)


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
        self.model = self.model.load_from_checkpoint(self.config.model_path, strict=False)

        self.pool_size = multiprocessing.cpu_count()
        if self.config.cuda:
            self.device_name = 'cuda'
        else:
            self.device_name = 'cpu'
        self.eye = torch.eye(self.max_length, device=self.device_name)
        self.zero = torch.zeros([self.max_length, self.max_length], device=self.device_name)

        if self.config.cuda:
            self.model = self.model.cuda()
            self.pool_size = int(np.floor(torch.cuda.get_device_properties('cuda').total_memory / (1024 ** 2) / 1968))
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
    
    def get_function_set_embedding(self, function_list: List[dict]):
        function_set: Dict[str, FunctionEmbedding] = {}
        for figure in function_list:
            name, embedding = figure['name'], FunctionEmbedding(name=figure['name'], embedding=figure['embedding'])
            function_set[name] = embedding
        return function_set
    
    def get_function_pool_embedding(self, function_list: Dict[str, List[dict]]) -> Tuple[list, List[FunctionEmbedding]]:
        function_result_list: List[FunctionEmbedding] = []
        name_list = []
        for function_name in function_list:
            for function_body in function_list[function_name]:
                name, embedding = function_body['name'], FunctionEmbedding(name=function_body['name'], embedding=function_body['embedding'])
                function_result_list.append(embedding)
                name_list.append(name)
        return name_list, function_result_list
    
    def get_single_function_embedding(self, dicts: dict) -> Tuple[str, FunctionEmbedding]:
        with torch.no_grad():
            tfeature, tadj, name = self.to_tensor(dicts)
            tembedding = self.model.my_model(tadj, tfeature)
            tembedding = tembedding.detach().cpu()
            
            embedding = tembedding.numpy()
            del tembedding
            
            # adj = tadj.detach().cpu().clone().numpy()
            del tadj
            
            # feature = tfeature.detach().cpu().clone().numpy()
            del tfeature

            return name, FunctionEmbedding(name=name, embedding=embedding)

    def to_tensor(self, json_dict: dict):
        adj = json_dict['adj']
        feature = json_dict['feature']
        name = json_dict['name']


        feature = torch.tensor(feature, dtype=torch.float, device=self.device_name)
        # print(feature.shape)
        adj_matrix = torch.zeros_like(self.zero, device=self.device_name)
        start = torch.tensor(adj[0], device=self.device_name)
        end = torch.tensor(adj[1], device=self.device_name)
        adj_matrix[start, end] = 1

        adj_matrix = (adj_matrix + adj_matrix.T)
        adj_matrix += self.eye

        # For padding
        feature_padder = torch.nn.ZeroPad2d([0, self.feature_length - feature.shape[1], 0, self.max_length - feature.shape[0]])
        # feature = feature_padder(feature)[:self.max_length][:, :self.feature_length]
        feature = feature_padder(feature)
        # print("After: ", feature.shape)

        return feature, adj_matrix, name
    
    def single_dgl_to_embedding(self, data: dgl.DGLGraph):
        with torch.no_grad():
            
            padding = self.max_length - data.num_nodes()
            data = dgl.add_nodes(data, padding)
            data = dgl.add_self_loop(data)
            
            if self.config.cuda:
                data = data.to('cuda')
            tembedding = self.model.my_model(data)
            tembedding = tembedding.detach().cpu()
            
            embedding = tembedding.numpy()
            del tembedding
            
            return embedding
        
    
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
    
    # @profile
    def test_recall_K_pool(self, dataset:dict, graph: list, max_k=1000, cache_path=""):
        recall = {x: [] for x in range(1, max_k + 1)}
        self.config.topK = max_k
            
        record_total = {x: [0, 0] for x in range(1, max_k + 1)}
        dataset = self.merge_dgl_dict(dataset, graph)
        
        pbar = tqdm(total=len(dataset['data']))
        pbar.set_description("Testing Recall")
        for binary in dataset['data']:

            function_list1 = self.get_function_name_list(dataset['data'][binary])
            function_pool1 = dataset['data'][binary]
            with torch.no_grad():
                # with open("result.pkl", 'rb') as f:

                result = self.get_test_pairs_pool_embedding(function_list1=function_list1, function_pool=function_pool1, use_cache=cache_path)
                pbar.update()
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
    
    def test_recall_K_file_parallel_reduce(self, queue: multiprocessing.Queue, max_k: int):

        # recall_total = {key: [0, 0] for key in range(1, max_k + 1)}
        recall = {key: [] for key in range(1, max_k + 1)}

        while True:
            try:
                res = queue.get(timeout=1)
            except multiprocessing.TimeoutError:
                continue

            if isinstance(res, str):
                queue.put(recall)
                break

            assert isinstance(res, dict)
            for k in range(1, max_k + 1):
                print(res)
                recall[k].append(res[k][0] / res[k][1])

        # return recall

    
    def test_recall_K_file_parallel_map(self, dataset: dict, max_k, binary_name: str, update_callback: Callable, queue: multiprocessing.Queue):

        recall_total = {key: [0, 0] for key in range(1, max_k + 1)}

        print("Generating Function Pool for {}".format(binary_name))
        candidate_pool, candidate_name_list = self.get_function_file_set(dataset=dataset, binary_name=binary_name)
        # return 
        function_list1 = dataset['data'][binary_name].keys()
        for function_name in function_list1:
            for function_body in dataset['data'][binary_name][function_name]:
                update_callback()
                arch, opt = function_body['arch'], function_body['opt']
                
                if (arch, opt) not in candidate_pool:
                    continue
                
                name, query_embedding = self.get_single_function_embedding(function_body)
                query_embedding: FunctionEmbedding
                query_embedding = query_embedding.embedding
                
                mat2 = []
                for c in candidate_pool[(arch, opt)]:
                    c: FunctionEmbedding
                    mat2.append(c.embedding)
                mat2 = np.vstack(mat2)
                
                mm = get_cos_similar_multi(query_embedding, mat2)
                rank_list = sorted(zip(mm.reshape(-1), candidate_name_list[(arch, opt)]), key=lambda x: x[0], reverse=True)[:self.config.topK]
                for k in range(1, max_k + 1):
                    is_correct = self.judge(name, [x[1] for x in rank_list[:k]])
                    recall_total[k][0] += int(is_correct)
                    recall_total[k][1] += 1

                
                    
        queue.put(recall_total)
        
    
    def test_recall_K_file_parallel(self, dataset: dict, max_k: int = 10):
        self.config.topK = max_k
        pbar = tqdm(total=self.get_dataset_function_num(dataset))
        
        message_queue = queue.Queue(10)
        
        pool2 = ThreadPool(1)
        res = pool2.apply_async(self.test_recall_K_file_parallel_reduce, args=(message_queue, max_k))

        # pool = multiprocessing.Pool(self.pool_size)
        pool = ThreadPool(self.pool_size)
        for binary in dataset['data']:
            pool.apply_async(self.test_recall_K_file_parallel_map, args=(dataset, max_k, binary, pbar.update, message_queue))

        pool.close()
        pool.join()
            
        message_queue.put("end")
        pool2.close()
        pool2.join()
        while not message_queue.empty():
            print("Remaining Content:", message_queue.get())
        # recall_avg = message_queue.get()

        print("recall_avg", 0, '\n')
    
    def merge_dgl_dict(self, dataset: dict, graphs: List[dgl.DGLGraph]):
        pbar = tqdm(total=self.get_dataset_function_num(dataset))
        pbar.set_description("Converting DGL to Embedding")
        
        for binary in dataset['data']:
            for function_name in dataset['data'][binary]:
                for i in range(len(dataset['data'][binary][function_name])):
                    index = dataset['data'][binary][function_name][i]['index']
                    embedding = self.single_dgl_to_embedding(graphs[index])
                    dataset['data'][binary][function_name][i]['embedding'] = embedding
                    pbar.update()
        return dataset
    
    # @profile
    def test_recall_K_file(self, dataset:dict, graph: list, max_k: int = 10):
        recall = {x: [] for x in range(1, max_k + 1)}
        self.config.topK = max_k
        
        dataset = self.merge_dgl_dict(dataset, graph)
                
        pbar = tqdm(total=self.get_dataset_function_num(dataset))
        
        record_total = {x: [0, 0] for x in range(1, max_k + 1)}
        
        for binary in dataset['data']:
            print("Generating Function Pool for {}".format(binary))
            candidate_pool, candidate_name_list = self.get_function_file_set(dataset=dataset, binary_name=binary)
            # return 
            function_list1 = dataset['data'][binary].keys()
            for function_name in function_list1:
                for function_body in dataset['data'][binary][function_name]:
                    pbar.update()
                    arch, opt = function_body['arch'], function_body['opt']
                    
                    if (arch, opt) not in candidate_pool:
                        continue
                    
                    name, query_embedding = function_body['name'], FunctionEmbedding(name=function_body['name'], embedding=function_body['embedding'])
                    query_embedding: FunctionEmbedding
                    query_embedding = query_embedding.embedding
                    
                    mat2 = []
                    for c in candidate_pool[(arch, opt)]:
                        c: FunctionEmbedding
                        mat2.append(c.embedding)
                    mat2 = np.vstack(mat2)
                    
                    # mm = get_cos_similar_multi(query_embedding, mat2)
                    # mm = similarity_score(query_embedding, mat2)
                    mm = get_pearson_score(query_embedding, mat2)
                    rank_list = sorted(zip(mm.reshape(-1), candidate_name_list[(arch, opt)]), key=lambda x: x[0], reverse=True)[:self.config.topK]
                    for k in range(1, max_k + 1):
                        is_correct = self.judge(name, [x[1] for x in rank_list[:k]])
                        record_total[k][0] += int(is_correct)
                        record_total[k][1] += 1

            for k in range(1, max_k + 1):
                recall[k].append(record_total[k][0] / record_total[k][1])
            # return 
        
        avg_recall = []
        recall_avg = []
        for k in range(1, max_k + 1):
            avg_recall.append(record_total[k][0] / record_total[k][1])
            recall_avg.append(np.mean(recall[k]))
            
        print("avg_recall", avg_recall, '\n', "recall_avg", recall_avg, '\n')
                    
    # @profile
    def get_function_file_set(self, dataset: dict, binary_name) -> Tuple[Dict[tuple, List[FunctionEmbedding]], Dict[tuple, List[str]]]:
        candidate_pool: Dict[tuple, List[FunctionEmbedding]] = {}
        candidate_name_pool: Dict[tuple, List[str]] = {}
        count = 0
        for function_name in dataset['data'][binary_name]:
            for function_body in dataset['data'][binary_name][function_name]:
                arch, opt = function_body['arch'], function_body['opt']
                if (arch, opt) not in candidate_pool:
                    candidate_pool[(arch, opt)] = []
                    candidate_name_pool[(arch, opt)] = []
                    
                name, embedding = function_body['name'], FunctionEmbedding(name=function_body['name'], embedding=function_body['embedding'])
                candidate_pool[(arch, opt)].append(embedding)
                candidate_name_pool[(arch, opt)].append(name)
                count += 1
                # if count >= 5:
                #     return candidate_pool, candidate_name_pool
        
        return candidate_pool, candidate_name_pool
 
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

                                function_list1 = self.get_function_name_list(dataset['data'][binary_name])
                                function_list2 = self.get_function_name_list(dataset['data'][binary_name])
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

                                function_list1 = self.get_function_name_list(dataset['data'][binary_name])
                                function_list2 = self.get_function_name_list(dataset['data'][binary_name])
                                with torch.no_grad():
                                    score, label = self.ROC_pair(function_list1=function_list1, function_list2=function_list2)
                                    
                                auc_score.append(roc_auc_score(y_true=label, y_score=score))
                                
                            except ValueError:
                                print("No that Architecture and Opt_level")

        auc_score = np.mean(auc_score)
        return auc_score
        
    def get_dataset_function_num(self, dataset: dict):
        num = 0
        for binary_name in dataset['data']:
            for function_name in dataset['data'][binary_name]:
                num += len(dataset['data'][binary_name][function_name])
        return num
        
if __name__ == '__main__':

    # multiprocessing.set_start_method(method='forkserver', force=True)

    model_config = ModelConfig()
    
    with open("dataset/uboot_dataset/index_test_data_5.pkl", 'rb') as f:
        dataset = pickle.load(f)
        f.close()
        # bad_binary_list = []
        # for binary in dataset['data']:
        #     if len(dataset['data'][binary]) < 50:
        #         bad_binary_list.append(binary)
        # for binary in bad_binary_list:
        #     del dataset['data'][binary]
    
    graphs, _ = dgl.load_graphs("dataset/uboot_dataset/dgl_graphs.dgl")

    model_config.model_path = "lightning_logs/uboot_pearson2/checkpoints/last.ckpt"
    model_config.dataset_path = ""
    model_config.feature_length = 151
    model_config.max_length = 1000
    model_config.cuda = True
    model_config.topK = 50
    model = InferenceModel(model_config)
    
    # model.AUC_average(dataset)
    res = model.test_recall_K_file(dataset, graphs, max_k=model_config.topK)
    # res = model.test_recall_K_pool(dataset, graphs, max_k=10)
    
    # with open("./recall_allstar.pkl", 'wb') as f:
    #     pickle.dump(res, f)
    #     f.close()
    
    # data = [res[i] for i in [1, 5, 10, 20, 30, 40, 50]]
    # label = [str(x) for x in [1, 5, 10, 20, 30, 40, 50]]
    # plt.boxplot(data, labels=label)
    # plt.savefig("recall_allstar.png")