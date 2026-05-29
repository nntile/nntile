#!/usr/bin/env bash
# Run only those C++ (CTest) tests whose corresponding sources were updated
# in the current PR, similar to run-dirty-py-tests.sh for pytest.

set -e

if [ -d build/nntile/tests ]; then
    "$(dirname "$0")/restore-ctest-execute-bits.sh" build/nntile/tests
fi

branch=$1
base_branch=${2:-main}
ctest_label=${3:-}
if [ -z "$branch" ]; then
    branch=$(git branch --show-current)
    echo "no branch specified: assume current branch is $branch"
fi

ctest_label_args=()
case "$ctest_label" in
    core|graph)
        ctest_label_args=(-L "$ctest_label")
        echo ":: CTest label filter: ${ctest_label}"
        ;;
    "")
        ;;
    *)
        echo "Unknown ctest label filter: ${ctest_label}" >&2
        exit 2
        ;;
esac

echo ":: Diff base ${base_branch}..${branch}"
all_changed=$(git diff --name-only "${base_branch}..${branch}")

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
        include/nntile/defs.h.in | include/nntile.hh | include/nntile/core.hh)
            run_all=true; break ;;
        include/nntile/starpu.hh | include/nntile/starpu/config.hh)
            run_all=true; break ;;
        nntile/src/kernel/cblas.cc | nntile/src/kernel/cublas.cc)
            run_all=true; break ;;
        nntile/src/runtime.cc | nntile/src/tensor/graph_data_node.cc)
            run_all=true; break ;;
        nntile/tests/model/llama/generate_test_data.py)
            run_all=true; break ;;
        nntile/tests/model/gpt2/generate_test_data.py)
            run_all=true; break ;;
        nntile/tests/model/gptneo/generate_test_data.py)
            run_all=true; break ;;
        nntile/tests/model/t5/generate_test_data.py)
            run_all=true; break ;;
        nntile/tests/model/roberta/generate_test_data.py)
            run_all=true; break ;;
    esac
done <<< "$all_changed"

if $run_all; then
    echo ":: Core files changed, running all C++ tests"
    ctest --test-dir build -E wrappers -LE "(MPI|NotImplemented)" \
        "${ctest_label_args[@]}" --output-on-failure
    exit
fi

declare -A affected

# ---------- helper functions for layer propagation -------------------------
add_all_layers() {
    local op=$1
    for p in tests_core_kernel tests_core_starpu tests_core_tile \
             tests_graph_tile_ops tests_graph_tensor_ops; do
        affected["${p}_${op}"]=1
    done
}

add_from_starpu() {
    local op=$1
    for p in tests_core_starpu tests_core_tile tests_graph_tile_ops \
             tests_graph_tensor_ops; do
        affected["${p}_${op}"]=1
    done
}

add_from_tile() {
    local op=$1
    for p in tests_core_tile tests_graph_tile_ops tests_graph_tensor_ops; do
        affected["${p}_${op}"]=1
    done
}

add_from_tensor() {
    local op=$1
    affected["tests_graph_tensor_ops_${op}"]=1
}

# GPT-Neo graph model: run all block tests when shared code changes.
add_gptneo_model_tests() {
    for t in gptneo_config gptneo_mlp gptneo_attention gptneo_decoder \
             gptneo_model gptneo_causal; do
        affected["tests_graph_model_${t}"]=1
    done
}

add_t5_model_tests() {
    for t in t5_config t5_ff t5_attention t5_cross_attention \
             t5_encoder_block t5_decoder_block t5_model t5_conditional; do
        affected["tests_graph_model_${t}"]=1
    done
}

add_bert_model_tests() {
    affected["tests_graph_model_bert_config"]=1
    for t in bert_intermediate bert_attention bert_layer \
             bert_embeddings bert_model bert_mlm; do
        affected["tests_graph_model_${t}"]=1
        affected["tests_graph_model_${t}_data_setup"]=1
    done
}

# RoBERTa graph tests import safetensor layout helpers from
# nntile/tests/model/bert/generate_test_data.py only (not other BERT sources).
add_roberta_model_tests() {
    affected["tests_graph_model_roberta_config"]=1
    for t in roberta_intermediate roberta_attention roberta_layer \
             roberta_embeddings roberta_model roberta_mlm; do
        affected["tests_graph_model_${t}"]=1
        affected["tests_graph_model_${t}_data_setup"]=1
    done
}

# ---------- classify every changed file ------------------------------------
while IFS= read -r file; do
    [ -z "$file" ] && continue

    case "$file" in
        # ---- test files: run the specific test ----------------------------
        nntile/tests/constants.cc)
            affected["tests_nntile_constants"]=1 ;;
        nntile/tests/kernel/*.cc)
            affected["tests_core_kernel_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/starpu/*.cc)
            affected["tests_core_starpu_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/core/*.cc)
            affected["tests_core_tile_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/tensor/ops/*.cc)
            affected["tests_graph_tensor_ops_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/tensor/*.cc)
            affected["tests_graph_tensor_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/tile/ops/*.cc)
            affected["tests_graph_tile_ops_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/tile/*.cc)
            affected["tests_graph_tile_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/nn/ops/*.cc)
            affected["tests_graph_nn_ops_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/nn/*.cc)
            affected["tests_graph_nn_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/module/*.cc)
            affected["tests_graph_module_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/io/*.cc)
            affected["tests_graph_io_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/llama/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/gpt2/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/gptneo/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/t5/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/roberta/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/test_gptneo_fixture_helpers.hh)
            add_gptneo_model_tests ;;
        nntile/tests/model/test_t5_fixture_helpers.hh)
            add_t5_model_tests ;;
        nntile/tests/model/bert/generate_test_data.py)
            add_bert_model_tests
            add_roberta_model_tests ;;
        nntile/tests/*.cc)
            affected["tests_graph_$(basename "$file" .cc)"]=1 ;;

        # ---- kernel sources / headers → all layers -----------------------
        nntile/src/kernel/*/cpu.cc | nntile/src/kernel/*/cuda.cc | nntile/src/kernel/*/cuda.cu)
            add_all_layers "$(basename "$(dirname "$file")")" ;;
        include/nntile/kernel/*/cpu.hh | include/nntile/kernel/*/cuda.hh)
            add_all_layers "$(basename "$(dirname "$file")")" ;;
        include/nntile/kernel/*.hh)
            add_all_layers "$(basename "$file" .hh)" ;;

        # ---- starpu sources / headers → from starpu up -------------------
        nntile/src/starpu/*.cc)
            add_from_starpu "$(basename "$file" .cc)" ;;
        include/nntile/starpu/*.hh)
            add_from_starpu "$(basename "$file" .hh)" ;;

        # ---- tile sources / headers → from tile up -----------------------
        nntile/src/core/*.cc)
            add_from_tile "$(basename "$file" .cc)" ;;
        include/nntile/core/*.hh)
            add_from_tile "$(basename "$file" .hh)" ;;

        # ---- tensor sources / headers → from tensor up -------------------
        nntile/src/tensor/*.cc)
            add_from_tensor "$(basename "$file" .cc)" ;;
        include/nntile/tensor/*.hh)
            add_from_tensor "$(basename "$file" .hh)" ;;

        # ---- graph-level: only the matching test --------------------------
        nntile/src/tensor/ops/*.cc)
            affected["tests_graph_tensor_ops_$(basename "$file" .cc)"]=1 ;;
        include/nntile/tensor/ops/*.hh)
            affected["tests_graph_tensor_ops_$(basename "$file" .hh)"]=1 ;;
        nntile/src/tensor/*.cc)
            affected["tests_graph_tensor_$(basename "$file" .cc)"]=1 ;;
        include/nntile/tensor/*.hh)
            affected["tests_graph_tensor_$(basename "$file" .hh)"]=1 ;;
        nntile/src/tile/ops/*.cc)
            affected["tests_graph_tile_ops_$(basename "$file" .cc)"]=1 ;;
        include/nntile/tile/ops/*.hh)
            affected["tests_graph_tile_ops_$(basename "$file" .hh)"]=1 ;;
        nntile/src/tile/*.cc)
            affected["tests_graph_tile_$(basename "$file" .cc)"]=1 ;;
        include/nntile/tile/*.hh)
            affected["tests_graph_tile_$(basename "$file" .hh)"]=1 ;;
        nntile/src/nn/ops/*.cc)
            affected["tests_graph_nn_ops_$(basename "$file" .cc)"]=1 ;;
        include/nntile/nn/ops/*.hh)
            affected["tests_graph_nn_ops_$(basename "$file" .hh)"]=1 ;;
        nntile/src/nn/*.cc)
            affected["tests_graph_nn_$(basename "$file" .cc)"]=1 ;;
        include/nntile/nn/*.hh)
            affected["tests_graph_nn_$(basename "$file" .hh)"]=1 ;;
        nntile/src/module/*.cc)
            affected["tests_graph_module_$(basename "$file" .cc)"]=1 ;;
        include/nntile/module/*.hh)
            affected["tests_graph_module_$(basename "$file" .hh)"]=1 ;;
        nntile/src/io/*.cc)
            affected["tests_graph_io_$(basename "$file" .cc)"]=1 ;;
        include/nntile/io/*.hh)
            affected["tests_graph_io_$(basename "$file" .hh)"]=1 ;;
        nntile/src/model/llama/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        include/nntile/model/llama/*.hh)
            affected["tests_graph_model_$(basename "$file" .hh)"]=1 ;;
        nntile/src/model/gpt2/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        include/nntile/model/gpt2/*.hh)
            affected["tests_graph_model_$(basename "$file" .hh)"]=1 ;;
        nntile/src/model/gptneo/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        include/nntile/model/gptneo/*.hh)
            add_gptneo_model_tests ;;
        include/nntile/model/gptneo.hh)
            add_gptneo_model_tests ;;
        examples/gptneo_config_json.hh | examples/gptneo_generate.cc | examples/gptneo_graph_training.cc | examples/gptneo_generate.py)
            add_gptneo_model_tests ;;
        nntile/src/model/t5/*.cc)
            add_t5_model_tests ;;
        include/nntile/model/t5/*.hh)
            add_t5_model_tests ;;
        include/nntile/model/t5.hh)
            add_t5_model_tests ;;
        examples/t5_generate.cc | examples/t5_generate.py | examples/t5_graph_training.cc | examples/t5_config_json.hh | examples/prepare_tiny_seq2seq_train_bin.py | examples/run_t5_graph_training_demo.sh)
            add_t5_model_tests ;;
        nntile/src/dataset/seq2seq_lm_mmap.cc | include/nntile/dataset/seq2seq_lm_mmap.hh)
            add_t5_model_tests ;;
    esac
done <<< "$all_changed"

if [ ${#affected[@]} -eq 0 ]; then
    echo ":: Unknown changes (no pattern matched), running all C++ tests"
    ctest --test-dir build -E wrappers -LE "(MPI|NotImplemented)" \
        "${ctest_label_args[@]}" --output-on-failure
    exit
fi

# Split CI passes core|graph so each job only has that layer's tests in its
# build tree. Drop affected names from the other layer to avoid ctest exit 8
# ("No tests were found") when -R matches nothing in this tree.
if [ "$ctest_label" = core ] || [ "$ctest_label" = graph ]; then
    declare -A layer_affected
    for name in "${!affected[@]}"; do
        case "$name" in
            tests_"${ctest_label}"_*)
                layer_affected["$name"]=1
                ;;
        esac
    done
    unset affected
    declare -A affected
    for name in "${!layer_affected[@]}"; do
        affected["$name"]=1
    done
    if [ ${#affected[@]} -eq 0 ]; then
        echo ":: No ${ctest_label}-layer tests affected by this diff; skipping"
        exit 0
    fi
fi

# Build an anchored ctest regex.  The (_[0-9]+)? suffix accounts for
# multi-argument tests that get a numeric suffix (e.g. tests_core_tile_gemm_1).
patterns=$(printf '%s\n' "${!affected[@]}" | sort | paste -sd '|')
regex="^(${patterns})(_[0-9]+)?$"

echo ":: Running ${#affected[@]} affected C++ test pattern(s):"
printf '  - %s\n' "${!affected[@]}" | sort
echo ":: CTest regex: $regex"

ctest --test-dir build -R "$regex" -E wrappers -LE "(MPI|NotImplemented)" \
    "${ctest_label_args[@]}" --output-on-failure
