core_path=/home/damaoooo/Downloads/binutils-2.41/binutils
make_path=/home/damaoooo/Downloads/binutils-2.41
dest_path=/home/damaoooo/Downloads/plc_test/dataset/binutils

bin_path=${dest_path}/bin
lift_path=${dest_path}/lift
cpg_file_path=${dest_path}/cpg_file
c_dot_path=${cpg_file_path}/c_dot
c_json_path=${cpg_file_path}/c_json

joern_dir=/home/damaoooo/Downloads/joern-cli

function init_clean(){
    rm -rf ${dest_path}/*
    mkdir -p ${bin_path}
    mkdir -p ${lift_path}
    mkdir -p ${cpg_file_path}
    mkdir -p ${c_dot_path}
    mkdir -p ${c_json_path}
}

function compile(){
    cd ${make_path}
    opt_level=$1
    architecture=$2

    if [ ${architecture} == "gcc" ] 
    then
        host="--host=x86_64-linux-gnu"
    else
        host=--host=${architecture}
    fi

    flags="$1 -fno-inline -g -w"
    # git checkout v8.31
    # ./bootstrap
    make distclean
    rm ./*/config.cache
    ./configure ${host} "CFLAGS=${flags}"
    # cp ./getopt-cdefs.h.back lib/getopt-cdefs.h
    make -j33
}

function update_path(){
    opt_level=$1
    architecture=$2

    dest_path=/home/damaoooo/Project/plc/dataset/binutils/${architecture}${opt_level}

    mkdir -p ${dest_path}
    bin_path=${dest_path}/bin
    lift_path=${dest_path}/lift
    cpg_file_path=${dest_path}/cpg_file
    c_dot_path=${cpg_file_path}/c_dot
    c_json_path=${cpg_file_path}/c_json
}

function copy_file(){
    for i in ${core_path}/*
    do
    
        if [[ -x ${i} ]]; then
            # can't be end with .sh
            if [[ ${i} == *.sh ]]; then
                continue
            fi

            # can't be a directory
            if [[ -d ${i} ]]; then
                continue
            fi

            # filename can't be start with config
            filename=$(basename ${i})

            if [[ ${filename} == config* ]]; then
                continue
            fi

            if [[ ${filename} == *sysinfo ]]; then
                continue
            fi

            if [[ ${filename} == *libtool* ]]; then
                continue
            fi

            cp ${i} ${bin_path}
        fi
    done
}

function lift_and_cpg(){
    for i in ${bin_path}/*
    do
        cd ${bin_path}
        retdec-decompiler ${i} -s -k --backend-keep-library-funcs
        filename=$(basename ${i})
        llvm-dis ${filename}.bc -o ${filename}.ll
        clang -m32 -O3 -c ${i}.ll -fno-inline-functions -o ${lift_path}/${filename}_re
        cd ${lift_path}
        retdec-decompiler ${lift_path}/${filename}_re -s -k --backend-keep-library-funcs
        ${joern_dir}/joern-parse --language c ${lift_path}/${filename}_re.c -o ${lift_path}/${filename}.cpg
        ${joern_dir}/joern-export ${lift_path}/${filename}.cpg -o ${lift_path}/c_dot_${filename}
    done
}

function final_clean(){
    rm ${lift_path}/*.dsm
    rm ${lift_path}/*.config.json
    rm ${lift_path}/*.bc
    rm ${lift_path}/*.cpg
    rm ${lift_path}/*.ll
    rm ${lift_path}/*.c

    rm ${bin_path}/*.dsm
    rm ${bin_path}/*.config.json
    rm ${bin_path}/*.bc
    rm ${bin_path}/*.ll
    rm ${bin_path}/*.c
}

function merge(){

    cnt=1
    for dir_path in ${lift_path}/c_dot_*
    do
        for cpg_file in ${dir_path}/*
        do
            if [ "${cpg_file##*.}" = "dot" ]; then
                cp ${cpg_file} ${c_dot_path}/${cnt}.dot
                cnt=`expr ${cnt} + 1`
            fi
        done
    done
}

function convert(){
    /home/damaoooo/miniconda3/envs/ml/bin/python /home/damaoooo/Project/plc/convert_to_json.py --input ${c_dot_path} --output ${c_json_path}
}
# init_clean
# copy_file
# lift_and_cpg
# final_clean
# merge
# convert
for arch in arm-linux-gnueabi mips-linux-gnu powerpc-linux-gnu gcc
do
    for opt in -O0 -O1 -O2 -O3
    do
        compile ${opt} ${arch}
        update_path ${opt} ${arch}
        init_clean
        copy_file
        lift_and_cpg
        final_clean
        # merge
        # convert
    done
done
