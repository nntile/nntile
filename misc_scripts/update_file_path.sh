#!/bin/sh
#
# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file misc_scripts/update_file_path.sh
# Hook to update @file-paths of all files to match their actual paths
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-05-05

# Change directory to the root of the repository
cd $(git rev-parse --show-toplevel)
# Get all files
all_files=$(git ls-files)
# List of excludes
exclude="external/random.hh external/pybind11"
# Walk through files
for file in ${all_files}
do
    if echo ${exclude} | grep "${file}" -qv
    then
        pattern="@file.*$" 
        # sed fails with "/", need to make it "\/"
        subst=$(echo "${file}" | gsed 's/\//\\\//g')
        #echo "patern=@date ${today}"
        #echo "subst=${subst}"
        gsed "0,/${pattern}/{s/${pattern}/@file ${subst}/}" -i ${file}
    fi
done
