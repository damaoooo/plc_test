import json
import pygraphviz as pgv
import networkx as nx
import numpy as np
import os
import re
import json
import pickle
import html
import torch
import cxxfilt
import shutil
import argparse
import random
from multiprocessing import Pool, Queue
import multiprocessing
import threading

np.set_printoptions(suppress=True)


class Converter:
    def __init__(self):
        self.LABEL = ['IDENTIFIER', 'LITERAL', 'BLOCK',
                      'CONTROL_STRUCTURE', 'LOCAL', 'METHOD_RETURN',
                      'METHOD', 'FIELD_IDENTIFIER', 'UNKNOWN',
                      'RETURN', 'PARAM', 'JUMP_TARGET', 'CALL']
        self.OP = ['<operator>.assignment', '<operator>.assignmentMinus',
                   '<operator>.assignmentPlus', '<operator>.multiplication', '<operator>.cast',
                   '<operator>.subtraction', '<operator>.fieldAccess', '<operator>.lessThan', '<operator>.logicalNot',
                   '<operator>.postIncrement', '<operator>.addressOf', '<operator>.addition', '<operator>.logicalShiftRight',
                   '<operator>.equals', '<operator>.indirection', '<operator>.minus', "<operator>.shiftLeft",
                   '<operator>.notEquals', '<operator>.greaterEqualsThan', '<operator>.postDecrement',
                   '<operator>.logicalOr', '<operator>.division', '<operator>.logicalAnd', "<operator>.lessEqualsThan",
                   '<operator>.delete', '<operator>.and', '<operator>.greaterThan', '<operator>.modulo', "<operator>.select",
                   '<operator>.new', "<operator>.conditional", "<operator>.or", "<operator>.xor", "<operator>.arithmeticShiftRight"]
        self.FUNC = ['printLine', 'free', 'malloc', 'wcslen', 'strlen', 'CLOSE_SOCKET', 'strcpy',
                     'strtoul', 'wcscpy', 'printWLine', 'socket', 'recv', 'memset',
                     'htons', 'WSAStartup', 'WSACleanup', 'MAKEWORD', 'fgets', 'listen',
                     'bind', 'accept', 'inet_addr', 'connect', 'fscanf', 'rand',
                     'fgetws', 'globalReturnsTrue', 'fopen', 'fclose', 'GETENV',
                     'staticReturnsTrue', 'wcsncat', 'strncat', 'globalReturnsTrueOrFalse',
                     'globalReturnsFalse', 'staticReturnsFalse', 'strchr', 'printLongLongLine']
        self.TYPE = ['char', 'int', 'short', 'float', 'double', 'long', 'string',
                     'void', 'struct', 'union', 'signed', 'unsigned', '*', 'array', 'vector', 'map']
        self.LLVMTYPE = ['i1', 'i2', 'i4', 'i8', 'i16', 'i32', 'i64', 'i128']

        self.signTable = {}
        self.length = 0

    def load_op_list(self, path):
        with open(path, 'rb') as f:
            result = pickle.load(f)
            f.close()
        assert isinstance(result, list)
        self.OP = result
        self.length = (len(self.LABEL) + 2) + (len(self.OP) + 2) + (len(self.FUNC) + 2) + 32 + (len(self.TYPE) + 2)

    def save_op_list(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.OP, f)
            f.close()

    def cleanSignTable(self):
        self.signTable = {}

    def findSignTable(self, variable: str):
        if variable in self.signTable:
            return self.signTable[variable]
        else:
            return 'UNKNOWN'

    def getTypeValue(self, vType: str):
        res = [0] * (len(self.TYPE) + 2)
        unknown = 0
        for type in self.TYPE:
            if type in vType:
                res[self.TYPE.index(type)] = 1
                unknown = 1
        if unknown == 0:
            res[len(res) - 2] = 1
        return res

    def getLLVMTypeValue(self, vType: str):
        res = [0] * (len(self.TYPE) + 2)
        if '*' in vType:
            res[len(self.LLVMTYPE)] = 1
        vType = vType.replace('*', '')
        if vType in self.LLVMTYPE:
            res[self.LLVMTYPE.index(vType)] = 1
        else:
            res[len(self.LLVMTYPE) + 1] = 1
        return res

    def convert(self, data: list):

        labelVec = [0] * (len(self.LABEL) + 2)
        opVec = [0] * (len(self.OP) + 2)
        funcVec = [0] * (len(self.FUNC) + 2)
        liteVec = [0] * 32
        typeVec = [0] * (len(self.TYPE) + 2)

        label: str = data[0]
        if label in self.LABEL:
            labelVec[self.LABEL.index(label)] = 1

        if label == 'LOCAL':
            t: str = data[1]
            vName, vType = t.split(':')
            self.signTable[vName.strip()] = vType.strip()
            typeVec = self.getTypeValue(vType)

        elif label == 'IDENTIFIER':
            vName: str = data[1]  # Variable Name
            typeVec = self.getTypeValue(self.findSignTable(vName))

        elif label == 'LITERAL':
            uNum: str = data[1]  # unknown Number
            if '\'' in uNum:
                typeVec = self.getTypeValue('char')
                ch = [x for x in uNum if x not in ['\'', '\"', '\\']][0]
                ch: str
                assert len(ch) == 1
                liteVec = self.int2binVec(ord(ch), signed=False)

            elif '\"' in uNum:
                typeVec = self.getTypeValue('string')
                liteVec = self.int2binVec(len(uNum[1:-1]))

            elif '0x' in uNum or '0X' in uNum:
                try:
                    value = int(uNum, 16)
                    typeVec = self.getTypeValue('int' + 'unsigned')
                    liteVec = self.int2binVec(value)
                except ValueError:
                    print('UNKNOWN LITERAL {}'.format(uNum))

                    typeVec = self.getTypeValue('UNKNOWN')
            elif 'L' or 'l' in uNum:
                uNum = uNum.replace('L', "").replace('l', '')
                typeVec = self.getTypeValue('int'+'signed')

            elif '.' in uNum or 'f' in uNum.upper() or 'e' in uNum.lower():
                typeVec = self.getTypeValue('float' + 'signed')
                hexValue = self.ConvertToFloat(
                    float(uNum.upper().split('F')[0]))
                liteVec = self.int2binVec(int(hexValue, 16), signed=False)
            elif uNum.isnumeric() or uNum[1:].isnumeric():
                value = int(uNum[0])
                typeVec = self.getTypeValue('int' + 'signed')
                liteVec = self.int2binVec(value)

            else:
                print('UNKNOWN LITERAL {}'.format(uNum))
                typeVec = self.getTypeValue('UNKNOWN')

        elif '<operator>' in label:
            labelVec[self.LABEL.index('CALL')] = 1
            try:
                opVec[self.OP.index(label)] = 1
            except ValueError:
                raise ValueError(label)

        elif label in self.FUNC:
            labelVec[self.LABEL.index('CALL')] = 1
            funcVec[self.FUNC.index(label)] = 1

        elif label == 'METHOD_RETURN':
            tName = data[1]
            typeVec = self.getTypeValue(tName)

        elif label == 'PARAM':
            s = data[1]
            s = s.split(' ')
            s = [x for x in s if s != '']
            vName = s[-1]
            sType = ' '.join(s[:-1])
            self.signTable[vName] = sType
            typeVec = self.getTypeValue(sType)

        output = labelVec + opVec + funcVec + liteVec + typeVec
        self.length = len(output)
        return output

    def int2binVec(self, num: int, signed=True):
        if signed:
            num = (bin(((1 << 32) - 1) & num)[2:]).zfill(32)
        else:
            num = bin(num)[2:].zfill(32)
        num = [ord(x) - ord('0') for x in num]
        return num

    # 浮点数整数部分转换成补码(整数全部为正)
    def ConvertFixedIntegerToComplement(self, fixedInterger):
        return bin(fixedInterger)[2:]

    def ConvertFixedDecimalToComplement(self, fixedDecimal):  # 浮点数小数部分转换成补码
        fixedpoint = int(fixedDecimal) / (10.0 ** len(fixedDecimal))
        s = ''
        while fixedDecimal != 1.0 and len(s) < 23:
            fixedpoint = fixedpoint * 2.0
            s += ('%16f' % fixedpoint)[0]
            fixedpoint = fixedpoint if ('%16f' % fixedpoint)[
                0] == '0' else fixedpoint - 1.0
        return s

    def ConvertToExponentMarker(self, number):  # 阶码生成
        return bin(number + 127)[2:].zfill(8)

    def ConvertToFloat(self, floatingPoint: float):  # 转换成IEEE754标准的数
        floatingPointString = '%16f' % floatingPoint
        if floatingPointString.find('-') != -1:  # 判断符号位
            sign = '1'
            floatingPointString = floatingPointString[1:]
        else:
            sign = '0'
        l = floatingPointString.split('.')  # 将整数和小数分离
        front = self.ConvertFixedIntegerToComplement(int(l[0]))  # 返回整数补码
        rear = self.ConvertFixedDecimalToComplement(l[1])  # 返回小数补码
        floatingPointString = front + '.' + rear  # 整合
        relativePos = floatingPointString.find(
            '.') - floatingPointString.find('1')  # 获得字符1的开始位置
        if relativePos > 0:  # 若小数点在第一个1之后
            exponet = self.ConvertToExponentMarker(relativePos - 1)  # 获得阶码
            mantissa = floatingPointString[
                floatingPointString.find('1') + 1: floatingPointString.find('.')] + floatingPointString[
                floatingPointString.find(
                    '.') + 1:]  # 获得尾数
        else:
            exponet = self.ConvertToExponentMarker(relativePos)  # 获得阶码
            mantissa = floatingPointString[floatingPointString.find(
                '1') + 1:]  # 获得尾数
        mantissa = mantissa[:23] + '0' * (23 - len(mantissa))
        floatingPointString = '0b' + sign + exponet + mantissa
        return hex(int(floatingPointString, 2))


def convert_file(file_name: str, converter: Converter, queue: Queue, control: Queue):
    G = nx.Graph(pgv.AGraph(file_name))

    function_name = re.sub("_part_\d+", "", G.name)
    function_name = re.sub("_constprop_\d+", "", function_name)
    function_name = re.sub("_isra_\d+", "", function_name)
    try:
        function_name = cxxfilt.demangle(function_name)
    except cxxfilt.InvalidName:
        queue.put({})
        return {}

    # if not ("_init_" in function_name or "_body" in function_name):
    #     continue

    G = G.to_undirected()
    edges = G.edges
    start = [int(x[0]) for x in edges]
    end = [int(x[1]) for x in edges]
    base = min(start + end)
    start = [x - base for x in start]
    end = [x - base for x in end]
    adj = [start, end]
    # adj = np.array(nx.adjacency_matrix(G).todense()).tolist()

    if len(adj[0]) < 10 or len(adj[0]) > 1000:
        queue.put({})
        return

    features = []
    for nodes in G.nodes:
        s: str = G.nodes[nodes]['label']
        s = html.unescape(s)
        tpl = s[1:-1].split(',')
        try:
            features.append(converter.convert(tpl))
        except ValueError as e:
            control.put(e)
            queue.put({})
            return

    out = {'adj': adj, "feature": features, "name": function_name}
    queue.put(out)
    return


def print_sample(x):
    print(x['name'], x['archtecture'], x['binary'], x['opt_level'])


def purify(x):
    return x['adj'], x['feature']


def make_paris(data):
    new_pairs = {"data": [], "adj_len": data['adj_len'],
                 "feature_len": data['feature_len']}

    for binary in data['data']:
        for func in data['data'][binary]:
            for x in data['data'][binary][func]:
                # print(x['name'], x['archtecture'], x['binary'], x['opt_level'])
                try:
                    same = random.choice(
                        [xx for xx in data['data'][binary][func] if xx != x])
                except IndexError:
                    same = random.choice(
                        [xx for xx in data['data'][binary][func]])
                different_binary_name = random.choice(
                    list(data['data'].keys()))
                different_function_name = random.choice(
                    list(data['data'][different_binary_name].keys()))
                different = random.choice(
                    data['data'][different_binary_name][different_function_name])
                new_pairs['data'].append(
                    [purify(x), purify(same), purify(different)])
    return new_pairs


def slice_store(pairs, offset, path):

    file_list = []
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    cnt = 0

    for k in [pairs['data'][x*offset:(x+1)*offset] for x in range(len(pairs['data'])//offset + 1)]:
        with open(os.path.join(path, f"{cnt}.pkl"), 'wb') as f:
            pickle.dump(k, f)
            f.close()
        file_list.append(f"{cnt}.pkl")
        cnt += 1

    meta_info = {"adj_len": pairs['adj_len'], "feature_len": pairs['feature_len'], "length": len(
        pairs['data']), "offset": offset, "file_list": file_list}

    with open(os.path.join(path, "metainfo.json"), 'w') as f:
        f.write(json.dumps(meta_info))
        f.close()


def split_dataset_by_binary(dataset: dict, ratio=0.1):
    test_num = int(ratio * len(dataset['data']))
    test_binaries = random.sample(list(dataset['data'].keys()), test_num)

    total_function_num = 0

    count = 0
    for test_b in dataset['data']:
        for func in dataset['data'][test_b]:
            total_function_num += len(dataset['data'][test_b][func])
            if test_b in test_binaries:
                count += len(dataset['data'][test_b][func])

    print("Selected Binary File:", test_binaries)
    print("Test Function", count, "Total Function",
          total_function_num, "Ratio:", count / total_function_num)
    test_data = {
        'data': {}, "adj_len": dataset['adj_len'], "feature_len": dataset['feature_len']}

    for test_b in test_binaries:
        test_data['data'][test_b] = dataset['data'][test_b]
        del dataset['data'][test_b]

    return dataset, test_data


def split_dataset_by_name(dataset: dict, ratio=0.1, binary_name: str = 'uboot'):
    test_num = int(ratio * len(dataset['data'][binary_name]))
    test_functions = random.sample(
        list(dataset['data'][binary_name].keys()), test_num)

    count = 0
    total_function_num = 0
    for func in dataset['data'][binary_name]:
        total_function_num += len(dataset['data'][binary_name][func])
        if func in test_functions:
            count += len(dataset['data'][binary_name][func])

    print("Selected Function Name:", test_functions)
    print("Test Function", count, "Total Function",
          total_function_num, "Ratio:", count / total_function_num)
    test_data = {
        'data': {}, "adj_len": dataset['adj_len'], "feature_len": dataset['feature_len']}

    test_data['data'][binary_name] = {}

    for test_func in test_functions:
        test_data['data'][binary_name][test_func] = dataset['data'][binary_name][test_func]
        del dataset['data'][binary_name][test_func]

    return dataset, test_data


def split_dataset_by_function(dataset: dict, ratio=0.1):
    test_num = int(ratio * len(dataset['data']))
    test_functions = random.sample(list(dataset['data'].keys()), test_num)

    count = 0
    total_function_num = 0

    for func in dataset['data']:
        total_function_num += len(dataset['data'][func])
        if func in test_functions:
            count += len(dataset['data'][func])

    print("Selected Function Name:", test_functions)
    print("Test Function", count, "Total Function",
          total_function_num, "Ratio:", count / total_function_num)

    test_data = {
        "data": {}, "adj_len": dataset['adj_len'], "feature_len": dataset['feature_len']}
    for test_func in test_functions:
        test_data['data'][test_func] = dataset['data'][test_func]
        del dataset['data'][test_func]

    return dataset, test_data


def filter_useful_function(dataset: dict, threshold=2):
    # Delete function that contains less than 2 samples
    bad_function_list = []
    for func in dataset['data']:
        if len(dataset['data'][func]) < threshold:
            bad_function_list.append(func)

    for func in bad_function_list:
        del dataset['data'][func]

    return dataset


def save_tensor(tensor, path):
    torch.save(tensor, path)


def load_tensor(path):
    obj = torch.load(path)
    return obj


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        f.close()


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        f.close()
    return obj


def collector(queue: Queue, control: Queue, all_data: dict):
    feature_len = 0
    adj_len = 0
    while 1:
        if queue.empty():
            continue

        if not control.empty():
            break
        else:
            x = queue.get()
            if x == "finish":
                print("I Got An Finish, Stop")
                break

            else:
                if not x:
                    continue
                print("Got", x['name'], len(all_data['data']))
                function_name = x['name']
                adj_len = max(adj_len, len(x['feature']))
                feature_len = max(len(x['feature'][0]), feature_len)

                if function_name not in all_data['data']:
                    all_data['data'][function_name] = [x]
                else:
                    all_data['data'][function_name].append(x)

    print("Finish")


def run_it(pwd, save_dir, converter: Converter, queue: Queue, control: Queue, pool: Pool, read_cache: bool = True):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.mkdir(save_dir)

    json_dir = os.path.join(save_dir, "json_file")
    os.mkdir(json_dir)

    cnt = 1

    max_nodes = 0

    all_data = {'data': {}, 'adj_len': 0, 'feature_len': 0}

    threading.Thread(target=collector, args=(
        queue, control, all_data), name="Collector").start()

    for arch in os.listdir(pwd):

        if read_cache:
            all_data = load_pickle("all_data_cache.pkl")
            all_data['data'] = all_data['data']['uboot'] # FIXME: Only for uboot
            print("Read Finish")
            break

        if not control.empty():
            print("Stoping Enumerate the architecture")
            break

        arch: str
        if arch.startswith('cpg'):
            continue

        cpg_dir = os.path.join(pwd, arch, 'cpg')

        for dot_file in os.listdir(cpg_dir):
            file_path = os.path.join(cpg_dir, dot_file)
            pool.apply(func=convert_file, args=(
                file_path, converter, queue, control))

            if not control.empty():
                print("Stoping Enumerate the function")
                break

    queue.put("finish")
    if not control.empty():
        print("Stoping the program")
        return

    pool.close()
    pool.join()

    print(f'max_nodes-{max_nodes}, feature_length-{converter.length}')

    all_data['adj_len'] = max_nodes
    all_data['feature_len'] = converter.length

    ############################################################
    #           filter the useful function
    ############################################################

    all_data = filter_useful_function(all_data, threshold=2)

    total_function_num = 0
    for func in all_data['data']:
        total_function_num += len(all_data['data'][func])
    print("Total Functions:", total_function_num)

    print("Convert Finish, saving cache")        
    save_pickle(all_data, "all_data.pkl")

    ############################################################
    #                   split the file
    ############################################################
    print("Making total dataset")
    save_pickle(all_data, os.path.join(save_dir, "total.pkl"))

    ############################################################
    #                   split the dataset
    ############################################################
    print("Making split dataset")
    random.seed(114514)
    train_set, test_set = split_dataset_by_function(all_data, ratio=0.1)
    save_pickle(train_set, os.path.join(save_dir, "train_set.pkl"))
    save_pickle(test_set, os.path.join(save_dir, "test_set.pkl"))


def iterative_run(input_dir: str, save_dir: str, op_list: str, queue: Queue, control: Queue, pool: Pool, cpu_num: int = os.cpu_count()):

    should_finish = False
    print("Start to run......")
    while 1:
        try:
            converter = Converter()
            converter.load_op_list(op_list)

            # Clear the queue
            while not queue.empty():
                queue.get()

            # Clear the control queue
            while not control.empty():
                control.get()

            run_it(input_dir, save_dir, converter, queue,
                   control, pool, read_cache=True)
            if not control.empty():
                print("Stoping the program")
                raise ValueError(control.get())
            should_finish = True
        except ValueError as e:
            converter.OP.append(str(e))
            converter.save_op_list(op_list)
            print("Lost:", str(e), "Adding to op_list")
            # stop the pool
            pool.terminate()

            # restart the pool
            pool = Pool(processes=cpu_num)

        if should_finish:
            break


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--input", '-i', type=str,
                      help="the input dot directory", default="uboot_dataset")
    args.add_argument("--output", '-o', type=str,
                      help="The outout directory", default="uboot_dataset/cpg_file")
    args.add_argument("--jobs", '-j', type=int,
                      help="The number of jobs", default=os.cpu_count())
    args = args.parse_args()

    pwd = args.input
    save_dir = args.output

    op_list = "./op_list.pkl"
    pool = Pool(processes=args.jobs)
    queue = multiprocessing.Manager().Queue(maxsize=10000)
    control = multiprocessing.Manager().Queue(maxsize=5)

    iterative_run(pwd, save_dir, op_list, queue,
                  control, pool, cpu_num=args.jobs)
