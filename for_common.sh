source_dir=/home/damaoooo/Project/coreutils-8.32/src
dest_dir=/home/damaoooo/Project/common
joern_dir=/home/damaoooo/Downloads/joern-cli

cd ${dest_dir}
rm -rf ./*


for compiler in gcc arm-linux-gnueabi-gcc powerpc-linux-gnu-gcc mips-linux-gnu-gcc
do
    for optimize in -O0 -O1 -O2 -O3
    do
        mkdir -p ${dest_dir}/${compiler}${optimize}
        cd ${dest_dir}/${compiler}${optimize}
        for prog in cat cp ls mv pwd rm timeout id sync wc who whoami yes
        do 
            filename=./${prog}
            $compiler ${optimize} -fno-inline-functions -fno-inline -c -g -I ${source_dir}/../lib/ -I ${source_dir}/../ -I ${source_dir} -static ${source_dir}/${prog}.c -o ${filename}.o
            retdec-decompiler ${filename}.o -s -k --backend-keep-library-funcs
            # opt ${filename}.o.ll -O3 --disable-inlining -S -o ${filename}_re.ll
            clang -m32 -c -g ${filename}.o.ll -O3 -fno-inline-functions -fno-inline -o ${filename}_re
            retdec-decompiler ${filename}_re -s -k --backend-keep-library-funcs
            rm ./*.dsm ./*.bc ./*.config.json ./*.ll 
        done
        ${joern_dir}/joern-parse --language c ${filename}_re.c -o ./cpg.cpg
        ${joern_dir}/joern-export ./cpg.cpg -o ./c_dot
        python /home/damaoooo/Project/plc/convert_to_json.py --input ./c_dot --output ./c_cpg
    done
done
