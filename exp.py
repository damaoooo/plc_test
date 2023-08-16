from PreProcessor.DataGenerator import DataGenerator, DataGeneratorMultiProcessing
from PreProcessor.FileScanner import FileScanner
from PreProcessor.GraphConverter import Converter

from line_profiler import LineProfiler

import pickle

if __name__ == '__main__':
    file_tree = FileScanner(root_path='coreutil_dataset').scan()
    converter = Converter(op_file=r'coreutil_dataset/op_file.pkl', read_op=True)
    # data_generator = DataGenerator(file_tree=file_tree, save_path='/home/damaoooo/mini_core/', converter=converter, read_cache=False)
    multi_datagen = DataGeneratorMultiProcessing(file_tree=file_tree, save_path='coreutil_dataset', converter=converter, read_cache=False)
    # data_generator.run()
    multi_datagen.run()
