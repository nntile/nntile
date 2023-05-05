#!/bin/sh
#
# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file misc-scripts/check_date_today.sh
# Hook to check if date of all new and modified files is today
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-05-05

# Change directory to the root of the repository
cd $(git rev-parse --show-toplevel)
# Get all A(dded), C(opied), M(odified) or R(enamed) files
mod_files=$(git diff --cached --name-only --diff-filter=ACMR)
# Date in proper format
today=$(date "+%Y-%m-%d")

# Collect all the incorrect files
mod_files_wrong_date=$(grep "@date" -m 1 -H ${mod_files} | grep "@date ${today}" -v)

# Print info if previous grep returned nothing
if [ $? -eq 0 ]
then
    echo "Today: ${today}"
    echo "Files that need a fresh date:"
    echo "${mod_files_wrong_date}"
    exit 1
fi

