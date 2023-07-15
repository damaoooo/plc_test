import os
import json
import time
import shutil
import subprocess
import multiprocessing
import argparse

def copy_file():
    pwd = "/ibex/tmp/zhoul0e/allstar"

    parts = ['p1', 'p2', 'p3', 'p4']
    arches = ['amd64', 'armel', 'i386', 'mipsel']
    arch = arches[0]
    part = parts[0]
    target_place = '/ibex/tmp/zhoul0e/all/'
    
    # Stats
    
    arches_functions = {"amd64": [], "armel": [], "i386": [], "mipsel": []}

    for part in parts:
        for arch in arches:
            diretory = os.path.join(pwd, part, arch)
            for binary in os.listdir(diretory):
                if 'index.json' in os.listdir(os.path.join(diretory, binary)):
                    try:
                        with open(os.path.join(diretory, binary, "index.json"), 'r') as f:
                            index = json.loads(f.read().replace("\n", ""))
                            f.close()
                        if 'binaries' in index:
                            for binary_file in index['binaries']:
                                binary_name = binary_file['name']
                                arches_functions[arch].append(binary_name)
                                
                    except json.JSONDecodeError:
                        print(os.path.join(diretory, binary), "Index File Wrong")
                        
    function_union = set(arches_functions['amd64']) & set(arches_functions['armel']) & set(arches_functions['i386']) & set(arches_functions['mipsel'])
    function_union = list(function_union)
    
    for part in parts:
        for arch in arches:
            diretory = os.path.join(pwd, part, arch)
            target_arch_path = os.path.join(target_place, arch, "bin")
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
                                binary_name = binary_file['name']
                                if binary_name in function_union:
                                    os.chmod(binary_file_path, 0o666)
                                    shutil.copy(binary_file_path, target_arch_path)
                                
                    except json.JSONDecodeError:
                        print(os.path.join(diretory, binary), "Index File Wrong")

def initial_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)
    
def lift_and_recompile(binary_file, lift_path):
    # Decompile the file
    os.chdir(os.path.split(binary_file)[0])
    ret = subprocess.run(args=['retdec-decompiler', binary_file, "-s -k --backend-keep-library-funcs"])
    if ret.returncode != 0:
        raise ValueError(f"The file {binary_file} can't be lifted")
    
    # Recompile the file
    file_name = os.path.basename(binary_file)
    ret = subprocess.run(args=['clang', '-m32', '-O3', '-c', f'{binary_file}.ll', '-fno-inline-functions', '-o', f'{lift_path}/{file_name}_re'])
    if ret.returncode != 0:
        raise ValueError(f"The file {binary_file}.ll can't be recompiled")
    
    # Decompile again
    os.chdir(lift_path)
    ret = subprocess.run(args=["retdec-decompiler", f"{lift_path}/{file_name}_re", "-s" ,"-k" ,"--backend-keep-library-funcs"])
    if ret.returncode != 0:
        raise ValueError(f"The file {lift_path}/{file_name}_re can't be re-lifted")
    
    ret = subprocess.run(args=["joern-parse", '--language' ,'c' ,f'{lift_path}/{file_name}_re.c' ,'-o', f'{lift_path}/{file_name}.cpg'])
    if ret.returncode != 0:
        raise ValueError(f"The file {lift_path}/{file_name}_re.c can't be convert into cpg")
    
    ret = subprocess.run(args=["joern-export" , f'{lift_path}/{file_name}.cpg', '-o', f'{lift_path}/c_dot_{file_name}'])
    if ret.returncode != 0:
        raise ValueError(f"The file {lift_path}/{file_name}.cpg can't be convert into dot file")
           
def convert_file():
    path = '/ibex/tmp/zhoul0e/all'
    arches = ['amd64', 'armel', 'i386', 'mipsel']
    # TODO: MultiProcessPools
    for arch in arches:
        arch_path = os.path.join(path, arch)
        
        bin_path = os.path.join(arch_path, 'bin')
        lift_path = os.path.join(arch_path, 'lifted')
        cpg_path = os.path.join(arch_path, 'cpg_file')
        
        initial_directory(lift_path)
        initial_directory(cpg_path)
        
        for file in os.listdir(bin_path):
            file_path = os.path.join(bin_path, file)
            try:
                lift_and_recompile(file_path, lift_path)
        
            except ValueError as e:
                print(e)
                continue
            
            except KeyboardInterrupt:
                print("Exiting...")
                exit(1)
                
            except:
                print(f"Something wrong happens in binary {file_path}")
                continue

def pool_lift(file_path, lift_path):
    print(f"Start to lift {file_path}")
    try:
        lift_and_recompile(file_path, lift_path)
    except:
        pass
            
def pool_convert_file(arch: str, pool: multiprocessing.Pool):
    
    path = '/ibex/tmp/zhoul0e/all'
    arch_path = os.path.join(path, arch)
    bin_path = os.path.join(arch_path, 'bin')
    lift_path = os.path.join(arch_path, 'lifted')
    cpg_path = os.path.join(arch_path, 'cpg_file')
    
    initial_directory(lift_path)
    initial_directory(cpg_path)
    
    for file in os.listdir(bin_path):
        file_path = os.path.join(bin_path, file)
        try:
            pool.apply_async(pool_lift, (file_path, lift_path))
    
        except ValueError as e:
            print(e)
            continue
        
        except KeyboardInterrupt:
            print("Exiting...")
            exit(1)
            
        except:
            print(f"Something wrong happens in binary {file_path}")
            continue
                
    pool.close()
    pool.join()

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", "-a", type=str, default='amd64', help="the architecture should be one of ['amd64', 'armel', 'i386', 'mipsel']")
    args = parser.parse_args()
    pool = multiprocessing.Pool(16)
    time1 = time.time()
    # copy_file() 
    pool_convert_file(args.arch, pool)
    print("copy file use: {:.2}s".format(time.time() - time1))
    