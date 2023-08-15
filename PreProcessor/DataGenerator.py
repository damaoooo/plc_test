import os
import random
import pickle
from tqdm import tqdm

from .FileScanner import FileTree
from .GraphConverter import Converter


def purify_cpg_json(cpg_json: dict):
    return {"adj": cpg_json['adj'], "feature": cpg_json['feature']}


def split_train_test_set(all_data: dict, ratio=0.1):
    binary_list = list(all_data.keys())
    test_binary = random.sample(binary_list, int(len(binary_list) * ratio))

    test_data = {}
    train_data = {}

    for binary in all_data:
        if binary in test_binary:
            test_data[binary] = all_data[binary]
        else:
            train_data[binary] = all_data[binary]

    return train_data, test_data


def purity_dataset(all_data: dict):
    for binary in all_data:
        for function in all_data[binary]:
            temp = []
            for cpg_json in all_data[binary][function]:
                temp.append(purify_cpg_json(cpg_json))
            all_data[binary][function] = temp.copy()
    
    return all_data

def filter_dataset(all_data: dict):
    bad_funtion = []
    for binary in all_data:
        for function in all_data[binary]:
            if len(all_data[binary][function]) < 2:
                bad_funtion.append((binary, function))        
    
    for binary, function in bad_funtion:
        del all_data[binary][function]
        
    bad_binary = []
    for binary in all_data:
        if len(all_data[binary]) < 2:
            bad_binary.append(binary)
    
    for binary in bad_binary:
        del all_data[binary]
        
    return all_data


def save_pickle(obj, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)
        f.close()


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        obj = pickle.load(f)
        f.close()
    return obj


class DataGenerator:
    def __init__(self, file_tree: FileTree, converter: Converter, save_path: str, read_cache: bool = False):
        self.file_tree = file_tree
        self.converter = converter
        self.save_path = save_path
        self.read_cache = read_cache

    def operator_walker(self):
        missing_operator = set()
        for file_path, _, _, _ in tqdm(self.file_tree):
            for c_dot_file in os.listdir(file_path):
                full_path = os.path.join(file_path, c_dot_file)
                try:
                    self.converter.probe_operator_file(full_path)
                except UserWarning as e:
                    missing_operator.add(str(e))

        origin_operator = self.converter.OP
        now_operator = origin_operator + list(missing_operator)
        self.converter.OP = now_operator
        self.converter.save_op_list(self.converter.op_file)
        return list(missing_operator)

    def generate_all_data(self):
        all_data = {}
        for file_path, opt, arch, binary in tqdm(self.file_tree):
            if binary not in all_data:
                all_data[binary] = {}
            for c_dot_file in os.listdir(file_path):
                c_dot_filename = os.path.join(file_path, c_dot_file)
                cpg_json = self.converter.convert_file(c_dot_filename, binary_name=binary, opt=opt, arch=arch)

                if not cpg_json:
                    continue

                function_name = cpg_json['name']
                if function_name not in all_data[binary]:
                    all_data[binary][function_name] = []
                all_data[binary][function_name].append(cpg_json)

        return all_data

    def wrap_dataset(self, dataset):
        return {"data": dataset, "adj": self.converter.max_length, "feature_len": self.converter.length}

    def run(self):
        
        if self.converter.read_op:
            self.converter.load_op_list(self.converter.op_file)
        else:
            print("Start to scan operator...")
            missing = self.operator_walker()
            print("adding missing operator: ", missing, 'to OP')
            self.converter.load_op_list(self.converter.op_file)
        
        if self.read_cache:
            print("Reading Cached dataset")
            all_data = load_pickle(os.path.join(self.save_path, 'origin_data.pkl'))
        else:
            print("Generating dataset...")
            all_data = self.generate_all_data()
            
            print("Saving the original dataset...")
            save_pickle(all_data, os.path.join(self.save_path, 'origin_data.pkl'))

        # return
        print("Splitting dataset...")
        train_data, test_data = split_train_test_set(all_data)
        train_data = purity_dataset(train_data)
        all_data = purity_dataset(all_data)
        
        print("saving dataset...")
        save_pickle(self.wrap_dataset(all_data), os.path.join(self.save_path, 'all_data.pkl'))
        save_pickle(self.wrap_dataset(train_data), os.path.join(self.save_path, 'train_data.pkl'))
        save_pickle(self.wrap_dataset(test_data), os.path.join(self.save_path, 'test_data.pkl'))



