from PreProcessor.DataGenerator import DataGenerator
from PreProcessor.FileScanner import FileScanner
from PreProcessor.GraphConverter import Converter

from line_profiler import LineProfiler

import pickle

if __name__ == '__main__':
    file_tree = FileScanner(root_path='/home/damaoooo/mini_core').scan()
    converter = Converter(op_file=r'/home/damaoooo/mini_core/op_file.pkl', read_op=True)
    data_generator = DataGenerator(file_tree=file_tree, save_path='/home/damaoooo/mini_core/', converter=converter, read_cache=True).run()