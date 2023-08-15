from PreProcessor.DataGenerator import DataGenerator
from PreProcessor.FileScanner import FileScanner
from PreProcessor.GraphConverter import Converter

import pickle

if __name__ == '__main__':
    file_tree = FileScanner(root_path='coreutil_dataset').scan()
    converter = Converter(op_file='coreutil_dataset/op_file.pkl', read_op=False)
    DataGenerator(file_tree=file_tree, save_path='./dataset/new_core', converter=converter, read_cache=True).run()
