#! /bin/bash

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pushd ${script_dir}

if [ -z "$1" ] ; then
    output_file=../explainable_face_recognition_$(date +%Y%m%d-%H%M%S).tgz
else
    output_file="$1"
fi

if [ -z "$2" ] ; then
    tag=master
else
    tag="$2"
fi

echo git archive --format=tar.gz -o "$output_file" --prefix=explainable_face_recognition/ "$tag" .
git archive --format=tar.gz -o "$output_file" --prefix=explainable_face_recognition/ "$tag" .
cd $(dirname "${output_file}")
md5sum $(basename "${output_file}") > $(basename "${output_file}").md5
popd
