import json
import pygraphviz as pgv
import networkx as nx
import numpy as np
import os
import re
import json
import pickle
import html
import cxxfilt
import shutil

np.set_printoptions(suppress=True)


def deserialize(file: str):
    with open(file, 'r') as f:
        content = f.read()
        f.close()
    content = json.loads(content)
    return content


def convert_node(node: dict):
    node_type: str = node['label']
    property = node['properties']

    if node_type == 'LOCAL':
        attr1 = property['NAME'] + ':' + property['TYPE_FULL_NAME']
        res = (node_type, attr1)
    elif node_type == 'IDENTIFIER':
        attr1 = property["NAME"]
        res = (node_type, attr1)
    elif node_type == 'LITERAL':
        attr1 = property["CODE"]
        res = (node_type, attr1)
    elif node_type == 'CALL':
        res = (property["NAME"], "")
    elif node_type == 'METHOD_RETURN':
        res = (node_type, property["TYPE_FULL_NAME"])
    else:
        res = (node_type, "")
    return res


def graph(content: dict):
    g = nx.Graph(pgv.AGraph())
    for node in content['nodes']:
        id: int = node['id']
        tpl = convert_node(node)
        tpl = str(tpl)
        g.add_node(id, **{'label':tpl})

    for edge in content['edges']:
        g.add_edge(edge['start'], edge['end'])
    return g


class Converter:
    def __init__(self):
        self.LABEL = ['IDENTIFIER', 'LITERAL', 'BLOCK',
                      'CONTROL_STRUCTURE', 'LOCAL', 'METHOD_RETURN',
                      'METHOD', 'FIELD_IDENTIFIER', 'UNKNOWN',
                      'RETURN', 'PARAM', 'JUMP_TARGET', 'CALL']
        self.OP = ['<operator>.assignment', '<operator>.indirectIndexAccess',
                   '<operator>.assignmentPlus', '<operator>.multiplication', '<operator>.cast',
                   '<operator>.subtraction', '<operator>.fieldAccess', '<operator>.lessThan', '<operator>.logicalNot',
                   '<operator>.postIncrement', '<operator>.addressOf', '<operator>.addition',
                   '<operator>.equals', '<operator>.indirection', '<operator>.minus', "<operator>.shiftLeft",
                   '<operator>.notEquals', '<operator>.greaterEqualsThan', '<operator>.postDecrement',
                   '<operator>.logicalOr', '<operator>.division', '<operator>.logicalAnd', "<operator>.lessEqualsThan",
                   '<operator>.delete', '<operator>.and', '<operator>.greaterThan', '<operator>.modulo',
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

        self.signTable = {}
        self.length = 0

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
                hexValue = self.ConvertToFloat(float(uNum.upper().split('F')[0]))
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
            opVec[self.OP.index(label)] = 1

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

    def ConvertFixedIntegerToComplement(self, fixedInterger):  # 浮点数整数部分转换成补码(整数全部为正)
        return bin(fixedInterger)[2:]

    def ConvertFixedDecimalToComplement(self, fixedDecimal):  # 浮点数小数部分转换成补码
        fixedpoint = int(fixedDecimal) / (10.0 ** len(fixedDecimal))
        s = ''
        while fixedDecimal != 1.0 and len(s) < 23:
            fixedpoint = fixedpoint * 2.0
            s += ('%16f' % fixedpoint)[0]
            fixedpoint = fixedpoint if ('%16f' % fixedpoint)[0] == '0' else fixedpoint - 1.0
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
        relativePos = floatingPointString.find('.') - floatingPointString.find('1')  # 获得字符1的开始位置
        if relativePos > 0:  # 若小数点在第一个1之后
            exponet = self.ConvertToExponentMarker(relativePos - 1)  # 获得阶码
            mantissa = floatingPointString[
                       floatingPointString.find('1') + 1: floatingPointString.find('.')] + floatingPointString[
                                                                                           floatingPointString.find(
                                                                                               '.') + 1:]  # 获得尾数
        else:
            exponet = self.ConvertToExponentMarker(relativePos)  # 获得阶码
            mantissa = floatingPointString[floatingPointString.find('1') + 1:]  # 获得尾数
        mantissa = mantissa[:23] + '0' * (23 - len(mantissa))
        floatingPointString = '0b' + sign + exponet + mantissa
        return hex(int(floatingPointString, 2))


if __name__ == '__main__':
    pwd = 'door/c_cpg'
    save_dir = "./c_door"
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    os.mkdir(save_dir)

    cnt = 1
    converter = Converter()

    max_nodes = 0

    all_data = []

    for file in os.listdir(pwd):
        fileName = os.path.join(pwd, file)
        G = nx.Graph(pgv.AGraph(fileName))
        
        function_name = re.sub("_part_\d+", "", G.name)
        function_name = re.sub("_constprop_\d+", "", function_name)
        function_name = re.sub("_isra_\d+", "", function_name)
        function_name = cxxfilt.demangle(function_name)
        function_name = function_name[:function_name.find('(')]

        if not ("_init_" in function_name or "_body" in function_name):
            continue
        
        max_nodes = max(len(G.nodes), max_nodes)
        G = G.to_undirected()
        edges = G.edges
        start = [int(x[0]) for x in edges]
        end = [int(x[1]) for x in edges]
        base = min(start + end)
        start = [x - base for x in start]
        end = [x - base for x in end]
        adj = [start, end]
        # adj = np.array(nx.adjacency_matrix(G).todense()).tolist()
        
        if len(adj[0]) < 10:
            continue
        
        features = []
        for nodes in G.nodes:
            s: str = G.nodes[nodes]['label']
            s = html.unescape(s)
            tpl = s[1:-1].split(',')
            features.append(converter.convert(tpl))
        
        out = {'adj': adj, "feature": features, "name": function_name}
        with open(os.path.join(save_dir, str(cnt - 1) + '.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(out))
            f.close()
        all_data.append(out)
        cnt += 1

    print(f'max_nodes-{max_nodes}, feature_length-{converter.length}')
    with open(os.path.join(save_dir, "!total.pkl"), 'wb') as f:
        pickle.dump({"data": all_data, "adj_len": max_nodes, "feature_len": converter.length}, f)
        f.close()
