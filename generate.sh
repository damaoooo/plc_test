source_dir=/home/damaoooo/Downloads/OpenPLC_v3/webserver/core
dest_dir=/home/damaoooo/Project/plc/blink
joern_dir=/home/damaoooo/Downloads/joern-cli

cd ${dest_dir}
rm -rf Res0_*
cd ${source_dir}
rm -rf Res0_*
for compiler in g++ arm-linux-gnueabi-g++ powerpc-linux-gnu-g++ mips-linux-gnu-g++
do
    for optimize in -O0 -O1 -O2 -O3
    do
        filename=${dest_dir}/Res0_${compiler}${optimize}
        echo ${filename}
        if [ ${compiler} == 'g++' ]
        then
            $compiler -std=gnu++11 -m32 -I ${source_dir}/lib -c Res0.c ${optimize} -fno-inline-functions -lasiodnp3 -lasiopal -lopendnp3 -lopenpal -w -o ${filename}
        else
            $compiler -std=gnu++11 -I ${source_dir}/lib -c Res0.c ${optimize} -fno-inline-functions -lasiodnp3 -lasiopal -lopendnp3 -lopenpal -w -o ${filename}
        fi
        retdec-decompiler ${filename} -s -k --backend-keep-library-funcs
        opt ${filename}.ll -O3 --disable-inlining -S -o ${filename}_re.ll
        # clang ${filename}.ll -O3 -m32 -emit-llvm -fno-inline-functions -S -o ${filename}_re.ll 
        clang -m32 -c ${filename}_re.ll -fno-inline-functions -o ${filename}_re
        # ${joern_dir}/ghidra2cpg ${filename}_re -o ${filename}_re.cpg
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