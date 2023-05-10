#!/bin/sh
#
# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file misc_scripts/check_file_path.sh
# Hook to check if @file-paths of all files match their actual paths
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-05-10

# Change directory to the root of the repository
cd $(git rev-parse --show-toplevel)
# Get all files
all_files=$(git ls-files)
# List of excludes
exclude="external/random.hh external/pybind11"
# Collect all the incorrect files
files_wrong_path=""
# Walk through files
for file in ${all_files}
do
    if echo ${exclude} | grep "${file}" -qv
    then
        tmp_val="$(grep "@file" -m 1 -H ${file} | grep "@file ${file}" -v)"
        if ! [ -z "${tmp_val}" ]
        then
            files_wrong_path="${files_wrong_path}\n${tmp_val}"
        fi
    fi
done

if ! [ -z "${files_wrong_path}" ]
then
    echo "Files that need a proper @file"
    echo "${files_wrong_path}"
    exit 1
fi

