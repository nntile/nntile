#!/usr/bin/env bash
# Run only those C++ (CTest) tests whose corresponding sources were updated
# in the current PR, similar to run-dirty-py-tests.sh for pytest.

set -e

branch=$1
if [ -z "$branch" ]; then
    branch=$(git branch --show-current)
    echo "no branch specified: assume current branch is $branch"
fi

all_changed=$(git diff --name-only main..$branch)

if [ -z "$all_changed" ]; then
    echo ":: No files changed"
    exit 0
fi

# If core build infrastructure changed, every C++ test is potentially affected.
run_all=false
while IFS= read -r file; do
    case "$file" in
        *CMakeLists.txt | cmake_modules/* | external/*)
            run_all=true; break ;;
        include/nntile/defs.h.in | include/nntile/nntile.hh)
            run_all=true; break ;;
        include/nntile/starpu.hh | include/nntile/starpu/config.hh)
            run_all=true; break ;;
        src/kernel/cblas.cc | src/kernel/cublas.cc)
            run_all=true; break ;;
        src/graph/tile/graph_runtime.cc | src/graph/tensor/graph_data_node.cc)
            run_all=true; break ;;
        tests/graph/model/llama/generate_test_data.py)
            run_all=true; break ;;
    esac
done <<< "$all_changed"

if $run_all; then
    echo ":: Core files changed, running all C++ tests"
    ctest --test-dir build -E wrappers -LE "(MPI|NotImplemented)" \
        --output-on-failure
    exit
fi

declare -A affected

# ---------- helper functions for layer propagation -------------------------
add_all_layers() {
    local op=$1
    for p in tests_kernel tests_starpu tests_tile tests_tensor \
             tests_graph_tensor; do
        affected["${p}_${op}"]=1
    done
}

add_from_starpu() {
    local op=$1
    for p in tests_starpu tests_tile tests_tensor tests_graph_tensor; do
        affected["${p}_${op}"]=1
    done
}

add_from_tile() {
    local op=$1
    for p in tests_tile tests_tensor tests_graph_tensor; do
        affected["${p}_${op}"]=1
    done
}

add_from_tensor() {
    local op=$1
    for p in tests_tensor tests_graph_tensor; do
        affected["${p}_${op}"]=1
    done
}

# ---------- classify every changed file ------------------------------------
while IFS= read -r file; do
    [ -z "$file" ] && continue

    case "$file" in
        # ---- test files: run the specific test ----------------------------
        tests/constants.cc)
            affected["tests_constants"]=1 ;;
        tests/kernel/*.cc)
            affected["tests_kernel_$(basename "$file" .cc)"]=1 ;;
        tests/starpu/*.cc)
            affected["tests_starpu_$(basename "$file" .cc)"]=1 ;;
        tests/tile/*.cc)
            affected["tests_tile_$(basename "$file" .cc)"]=1 ;;
        tests/tensor/*.cc)
            affected["tests_tensor_$(basename "$file" .cc)"]=1 ;;
        tests/graph/tensor/*.cc)
            affected["tests_graph_tensor_$(basename "$file" .cc)"]=1 ;;
        tests/graph/nn/*.cc)
            affected["tests_graph_nn_$(basename "$file" .cc)"]=1 ;;
        tests/graph/module/*.cc)
            affected["tests_module_$(basename "$file" .cc)"]=1 ;;
        tests/graph/io/*.cc)
            affected["tests_io_$(basename "$file" .cc)"]=1 ;;
        tests/graph/model/llama/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        tests/graph/*.cc)
            affected["tests_graph_$(basename "$file" .cc)"]=1 ;;

        # ---- kernel sources / headers → all layers -----------------------
        src/kernel/*/cpu.cc | src/kernel/*/cuda.cc | src/kernel/*/cuda.cu)
            add_all_layers "$(basename "$(dirname "$file")")" ;;
        include/nntile/kernel/*/cpu.hh | include/nntile/kernel/*/cuda.hh)
            add_all_layers "$(basename "$(dirname "$file")")" ;;
        include/nntile/kernel/*.hh)
            add_all_layers "$(basename "$file" .hh)" ;;

        # ---- starpu sources / headers → from starpu up -------------------
        src/starpu/*.cc)
            add_from_starpu "$(basename "$file" .cc)" ;;
        include/nntile/starpu/*.hh)
            add_from_starpu "$(basename "$file" .hh)" ;;

        # ---- tile sources / headers → from tile up -----------------------
        src/tile/*.cc)
            add_from_tile "$(basename "$file" .cc)" ;;
        include/nntile/tile/*.hh)
            add_from_tile "$(basename "$file" .hh)" ;;

        # ---- tensor sources / headers → from tensor up -------------------
        src/tensor/*.cc)
            add_from_tensor "$(basename "$file" .cc)" ;;
        include/nntile/tensor/*.hh)
            add_from_tensor "$(basename "$file" .hh)" ;;

        # ---- graph-level: only the matching test --------------------------
        src/graph/tensor/*.cc)
            affected["tests_graph_tensor_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/tensor/*.hh)
            affected["tests_graph_tensor_$(basename "$file" .hh)"]=1 ;;
        src/graph/nn/*.cc)
            affected["tests_graph_nn_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/nn/*.hh)
            affected["tests_graph_nn_$(basename "$file" .hh)"]=1 ;;
        src/graph/module/*.cc)
            affected["tests_module_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/module/*.hh)
            affected["tests_module_$(basename "$file" .hh)"]=1 ;;
        src/graph/io/*.cc)
            affected["tests_io_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/io/*.hh)
            affected["tests_io_$(basename "$file" .hh)"]=1 ;;
        src/graph/model/llama/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/model/llama/*.hh)
            affected["tests_graph_model_$(basename "$file" .hh)"]=1 ;;
    esac
done <<< "$all_changed"

if [ ${#affected[@]} -eq 0 ]; then
    echo ":: Unknown changes (no pattern matched), running all C++ tests"
    ctest --test-dir build -E wrappers -LE "(MPI|NotImplemented)" \
        --output-on-failure
    exit
fi

# Build an anchored ctest regex.  The (_[0-9]+)? suffix accounts for
# multi-argument tests that get a numeric suffix (e.g. tests_tile_gemm_1).
patterns=$(printf '%s\n' "${!affected[@]}" | sort | paste -sd '|')
regex="^(${patterns})(_[0-9]+)?$"

echo ":: Running ${#affected[@]} affected C++ test pattern(s):"
printf '  - %s\n' "${!affected[@]}" | sort
echo ":: CTest regex: $regex"

ctest --test-dir build -R "$regex" -E wrappers -LE "(MPI|NotImplemented)" \
    --output-on-failure
