#!/bin/sh
#
# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file misc-scripts/update_date_today.sh
# Hook to update date of all new and modified files to today
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-05-05

# Change directory to the root of the repository
cd $(git rev-parse --show-toplevel)
# Get all A(dded), C(opied), M(odified) or R(enamed) files
mod_files=$(git diff --name-only --diff-filter=ACMR)
# Date in proper format
today=$(date "+%Y-%m-%d")
# Pattern
pattern="@date 20..-..-.."

# Change only the first occurance
gsed "0,/${pattern}/{s/${pattern}/@date ${today}/}" -i ${mod_files}
