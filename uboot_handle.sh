data_path=/home/damaoooo/Project/plc/uboot_dataset
joern_dir=/home/damaoooo/Downloads/joern-cli


function lift_and_cpg(){
    for arch in ${data_path}/*
    do

        bin_path=${arch}/bin
        lift_path=${arch}/lifted

        cpg_path=${arch}/cpg
        rm -rf ${cpg_path}

        rm -rf ${lift_path}
        mkdir ${lift_path}

        cd ${arch}/bin

        for file in ${arch}/bin/*
        do
            cd ${bin_path}
            retdec-decompiler ${file} -s -k --backend-keep-library-funcs
            filename=$(basename ${file})
            clang -m32 -O3 -c ${file}.ll -fno-inline-functions -o ${lift_path}/${filename}_re
            cd ${lift_path}
            retdec-decompiler ${lift_path}/${filename}_re -s -k --backend-keep-library-funcs
            ${joern_dir}/joern-parse --language c ${lift_path}/${filename}_re.c -o ${lift_path}/${filename}.cpg
            ${joern_dir}/joern-export ${lift_path}/${filename}.cpg -o ${cpg_path}
            final_clean ${lift_path}
            final_clean ${bin_path}
            cd ${bin_path}
            mv ${filename}.c ${lift_path}/${filename}.c
            mv ${filename}.ll ${lift_path}/${filename}.ll
        done

    done
}

function final_clean(){
    rm $1/*.dsm
    rm $1/*.config.json
    rm $1/*.bc
}

lift_and_cpg