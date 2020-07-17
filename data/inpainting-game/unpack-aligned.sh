#! /bin/bash

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pushd IJBC

for file in subj-*.tar.gz ; do
    IFS='.-' read -ra namearr <<< $file
    echo "Unpacking ${file} into IJBC/aligned/${namearr[1]}"
    tar xfz "${file}"
done
