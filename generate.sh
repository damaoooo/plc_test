source_dir=/home/damaoooo/Downloads/OpenPLC_v3/webserver/core
dest_dir=/home/damaoooo/Project/plc/dataset/openplc
joern_dir=/home/damaoooo/Downloads/joern-cli

cd ${dest_dir}
rm -rf ./*

cd ${source_dir}
rm -rf Res0_*

for compiler in g++ arm-linux-gnueabi-g++ powerpc-linux-gnu-g++ mips-linux-gnu-g++
do
    for optimize in -O0 -O1 -O2 -O3
    do
        mkdir -p ${dest_dir}/${compiler}${optimize}
        filename=${dest_dir}/${compiler}${optimize}/Res0_${compiler}${optimize}.o

        if [ ${compiler} == 'g++' ]
        then
            $compiler -std=gnu++11 -m32 -I ${source_dir}/lib -c Res0.c ${optimize} -fno-inline-functions -fno-inline -lasiodnp3 -lasiopal -lopendnp3 -lopenpal -w -o ${filename}
        else
            $compiler -std=gnu++11 -I ${source_dir}/lib -c Res0.c ${optimize} -fno-inline-functions -fno-inline -lasiodnp3 -lasiopal -lopendnp3 -lopenpal -w -o ${filename}
        fi
        retdec-decompiler ${filename} -s -k --backend-keep-library-funcs
        llvm-dis ${filename}.bc -o ${filename}.ll
        clang -m32 -O3 -c ${filename}.ll -fno-inline-functions -o ${filename}_re
        retdec-decompiler ${filename}_re -s -k --backend-keep-library-funcs

        ${joern_dir}/joern-parse --language c ${filename}_re.c -o ${dest_dir}/${compiler}${optimize}/cpg.cpg
        ${joern_dir}/joern-export ${dest_dir}/${compiler}${optimize}/cpg.cpg -o ${dest_dir}/${compiler}${optimize}/c_dot

        rm ${dest_dir}/${compiler}${optimize}/*.dsm
        rm ${dest_dir}/${compiler}${optimize}/*.config.json
        rm ${dest_dir}/${compiler}${optimize}/*.bc
        rm ${dest_dir}/${compiler}${optimize}/*.c
        rm ${dest_dir}/${compiler}${optimize}/*.ll
        rm ${dest_dir}/${compiler}${optimize}/cpg.cpg

    done
done
