source_dir=/home/damaoooo/Downloads/OpenPLC_v3/webserver/core
dest_dir=/home/damaoooo/Project/plc/multi_language
joern_dir=/home/damaoooo/Downloads/joern-cli

cd ${dest_dir}
rm -rf Res0_*
rf -rf c_cpg
rm -rf *.cpg
cd ${source_dir}
rm -rf Res0_*
for compiler in g++ arm-linux-gnueabi-g++ powerpc-linux-gnu-g++ mips-linux-gnu-g++
do
    for optimize in -O1 -O2 -O3
    do
        filename=${dest_dir}/Res0_${compiler}${optimize}
        echo ${filename}
        if [ ${compiler} == 'g++' ]
        then
            $compiler -std=gnu++11 -m32 -I ${source_dir}/lib -c Res0.c ${optimize} -fno-inline-functions -fno-inline -lasiodnp3 -lasiopal -lopendnp3 -lopenpal -w -o ${filename}
        else
            $compiler -std=gnu++11 -I ${source_dir}/lib -c Res0.c ${optimize} -fno-inline-functions -fno-inline -lasiodnp3 -lasiopal -lopendnp3 -lopenpal -w -o ${filename}
        fi
        retdec-decompiler ${filename} -s -k --backend-keep-library-funcs
        opt ${filename}.ll -O3 --disable-inlining -S -o ${filename}_re.ll
        # clang ${filename}.ll -O3 -m32 -emit-llvm -fno-inline-functions -S -o ${filename}_re.ll 
        clang -m32 -c ${filename}.ll -fno-inline-functions -o ${filename}_re
        retdec-decompiler ${filename}_re -s -k --backend-keep-library-funcs
        # ${joern_dir}/c2cpg.sh ${filename}_re.c -o ${filename}_re.cpg
        # ${joern_dir}/joern-export ${filename}_re.cpg -o ${filename}_re_dot
        # ${llvmcpg} ${filename}.ll --output=${filename}.zip --inline --inline-strings
    done
done
cd ${dest_dir}
rm *.dsm
rm *.config.json
rm *.bc

${joern_dir}/joern-parse --language c ./ -o ./cpg.cpg
${joern_dir}/joern-export ./cpg.cpg -o ${dest_dir}/c_cpg