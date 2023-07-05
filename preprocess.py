import os
import json
import shutil

def copy_file():

    pwd = "/ibex/tmp/zhoul0e/allstar"

    parts = ['p1', 'p2', 'p3', 'p4']
    arches = ['amd64', 'armel', 'i386', 'mipsel']
    arch = arches[0]
    part = parts[0]
    target_place = '/ibex/tmp/zhoul0e/all/'
    diretory = os.path.join(pwd, part, arch)

    for part in parts:
        for arch in arches:
            
            target_arch_path = os.path.join(target_place, arch)
            print("move file from {} to {}".format(diretory, target_arch_path))
            if not os.path.exists(target_arch_path):
                os.mkdir(target_arch_path)

            for binary in os.listdir(diretory):
                if 'index.json' in os.listdir(os.path.join(diretory, binary)):
                    try:
                        with open(os.path.join(diretory, binary, "index.json"), 'r') as f:
                            index = json.loads(f.read().replace("\n", ""))
                            f.close()
                        if 'binaries' in index:
                            for binary_file in index['binaries']:
                                binary_file_path = os.path.join(diretory, binary, binary_file['file'])
                                shutil.copy(binary_file_path, target_arch_path)
                                
                    except json.JSONDecodeError:
                        print(os.path.join(diretory, binary), "Index File Wrong")
           
def convert_file():
    # TODO: MultiProcessPools
    pass
         
copy_file()