#!/usr/bin/env bash
# Run only those C++ (CTest) tests whose corresponding sources were updated
# in the current PR, similar to run-dirty-py-tests.sh for pytest.

set -e

branch=$1
base_branch=${2:-main}
if [ -z "$branch" ]; then
    branch=$(git branch --show-current)
    echo "no branch specified: assume current branch is $branch"
fi

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
        include/nntile/core/defs.h.in | include/nntile.hh | include/nntile/core.hh)
            run_all=true; break ;;
        include/nntile/core/starpu.hh | include/nntile/core/starpu/config.hh)
            run_all=true; break ;;
        src/core/kernel/cblas.cc | src/core/kernel/cublas.cc)
            run_all=true; break ;;
        src/graph/runtime.cc | src/graph/tensor/graph_data_node.cc)
            run_all=true; break ;;
        tests/graph/model/llama/generate_test_data.py)
            run_all=true; break ;;
        tests/graph/model/gpt2/generate_test_data.py)
            run_all=true; break ;;
        tests/graph/model/gptneo/generate_test_data.py)
            run_all=true; break ;;
        tests/graph/model/t5/generate_test_data.py)
            run_all=true; break ;;
        tests/graph/model/roberta/generate_test_data.py)
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
    for p in tests_core_kernel tests_core_starpu tests_core_tile tests_core_tensor \
             tests_graph_tensor_ops; do
        affected["${p}_${op}"]=1
    done
}

add_from_starpu() {
    local op=$1
    for p in tests_core_starpu tests_core_tile tests_core_tensor tests_graph_tensor_ops; do
        affected["${p}_${op}"]=1
    done
}

add_from_tile() {
    local op=$1
    for p in tests_core_tile tests_core_tensor tests_graph_tensor_ops; do
        affected["${p}_${op}"]=1
    done
}

add_from_tensor() {
    local op=$1
    for p in tests_core_tensor tests_graph_tensor_ops; do
        affected["${p}_${op}"]=1
    done
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
# tests/graph/model/bert/generate_test_data.py only (not other BERT sources).
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
        tests/constants.cc)
            affected["tests_core_constants"]=1 ;;
        tests/core/kernel/*.cc)
            affected["tests_core_kernel_$(basename "$file" .cc)"]=1 ;;
        tests/core/starpu/*.cc)
            affected["tests_core_starpu_$(basename "$file" .cc)"]=1 ;;
        tests/core/tile/*.cc)
            affected["tests_core_tile_$(basename "$file" .cc)"]=1 ;;
        tests/core/tensor/*.cc)
            affected["tests_core_tensor_$(basename "$file" .cc)"]=1 ;;
        tests/graph/tensor/ops/*.cc)
            affected["tests_graph_tensor_ops_$(basename "$file" .cc)"]=1 ;;
        tests/graph/tensor/*.cc)
            affected["tests_graph_tensor_$(basename "$file" .cc)"]=1 ;;
        tests/graph/tile/ops/*.cc)
            affected["tests_graph_tile_ops_$(basename "$file" .cc)"]=1 ;;
        tests/graph/tile/*.cc)
            affected["tests_graph_tile_$(basename "$file" .cc)"]=1 ;;
        tests/graph/nn/ops/*.cc)
            affected["tests_graph_nn_ops_$(basename "$file" .cc)"]=1 ;;
        tests/graph/nn/*.cc)
            affected["tests_graph_nn_$(basename "$file" .cc)"]=1 ;;
        tests/graph/module/*.cc)
            affected["tests_graph_module_$(basename "$file" .cc)"]=1 ;;
        tests/graph/io/*.cc)
            affected["tests_graph_io_$(basename "$file" .cc)"]=1 ;;
        tests/graph/model/llama/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        tests/graph/model/gpt2/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        tests/graph/model/gptneo/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        tests/graph/model/t5/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        tests/graph/model/roberta/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        tests/graph/model/test_gptneo_fixture_helpers.hh)
            add_gptneo_model_tests ;;
        tests/graph/model/test_t5_fixture_helpers.hh)
            add_t5_model_tests ;;
        tests/graph/model/bert/generate_test_data.py)
            add_bert_model_tests
            add_roberta_model_tests ;;
        tests/graph/*.cc)
            affected["tests_graph_$(basename "$file" .cc)"]=1 ;;

        # ---- kernel sources / headers → all layers -----------------------
        src/core/kernel/*/cpu.cc | src/core/kernel/*/cuda.cc | src/core/kernel/*/cuda.cu)
            add_all_layers "$(basename "$(dirname "$file")")" ;;
        include/nntile/core/kernel/*/cpu.hh | include/nntile/core/kernel/*/cuda.hh)
            add_all_layers "$(basename "$(dirname "$file")")" ;;
        include/nntile/core/kernel/*.hh)
            add_all_layers "$(basename "$file" .hh)" ;;

        # ---- starpu sources / headers → from starpu up -------------------
        src/core/starpu/*.cc)
            add_from_starpu "$(basename "$file" .cc)" ;;
        include/nntile/core/starpu/*.hh)
            add_from_starpu "$(basename "$file" .hh)" ;;

        # ---- tile sources / headers → from tile up -----------------------
        src/core/tile/*.cc)
            add_from_tile "$(basename "$file" .cc)" ;;
        include/nntile/core/tile/*.hh)
            add_from_tile "$(basename "$file" .hh)" ;;

        # ---- tensor sources / headers → from tensor up -------------------
        src/core/tensor/*.cc)
            add_from_tensor "$(basename "$file" .cc)" ;;
        include/nntile/core/tensor/*.hh)
            add_from_tensor "$(basename "$file" .hh)" ;;

        # ---- graph-level: only the matching test --------------------------
        src/graph/tensor/ops/*.cc)
            affected["tests_graph_tensor_ops_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/tensor/ops/*.hh)
            affected["tests_graph_tensor_ops_$(basename "$file" .hh)"]=1 ;;
        src/graph/tensor/*.cc)
            affected["tests_graph_tensor_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/tensor/*.hh)
            affected["tests_graph_tensor_$(basename "$file" .hh)"]=1 ;;
        src/graph/tile/ops/*.cc)
            affected["tests_graph_tile_ops_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/tile/ops/*.hh)
            affected["tests_graph_tile_ops_$(basename "$file" .hh)"]=1 ;;
        src/graph/tile/*.cc)
            affected["tests_graph_tile_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/tile/*.hh)
            affected["tests_graph_tile_$(basename "$file" .hh)"]=1 ;;
        src/graph/nn/ops/*.cc)
            affected["tests_graph_nn_ops_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/nn/ops/*.hh)
            affected["tests_graph_nn_ops_$(basename "$file" .hh)"]=1 ;;
        src/graph/nn/*.cc)
            affected["tests_graph_nn_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/nn/*.hh)
            affected["tests_graph_nn_$(basename "$file" .hh)"]=1 ;;
        src/graph/module/*.cc)
            affected["tests_graph_module_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/module/*.hh)
            affected["tests_graph_module_$(basename "$file" .hh)"]=1 ;;
        src/graph/io/*.cc)
            affected["tests_graph_io_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/io/*.hh)
            affected["tests_graph_io_$(basename "$file" .hh)"]=1 ;;
        src/graph/model/llama/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/model/llama/*.hh)
            affected["tests_graph_model_$(basename "$file" .hh)"]=1 ;;
        src/graph/model/gpt2/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/model/gpt2/*.hh)
            affected["tests_graph_model_$(basename "$file" .hh)"]=1 ;;
        src/graph/model/gptneo/*.cc)
            affected["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        include/nntile/graph/model/gptneo/*.hh)
            add_gptneo_model_tests ;;
        include/nntile/graph/model/gptneo.hh)
            add_gptneo_model_tests ;;
        examples/gptneo_config_json.hh | examples/gptneo_generate.cc | examples/gptneo_graph_training.cc | examples/gptneo_generate.py)
            add_gptneo_model_tests ;;
        src/graph/model/t5/*.cc)
            add_t5_model_tests ;;
        include/nntile/graph/model/t5/*.hh)
            add_t5_model_tests ;;
        include/nntile/graph/model/t5.hh)
            add_t5_model_tests ;;
        examples/t5_generate.cc | examples/t5_generate.py | examples/t5_graph_training.cc | examples/t5_config_json.hh | examples/prepare_tiny_seq2seq_train_bin.py | examples/run_t5_graph_training_demo.sh)
            add_t5_model_tests ;;
        src/graph/dataset/seq2seq_lm_mmap.cc | include/nntile/graph/dataset/seq2seq_lm_mmap.hh)
            add_t5_model_tests ;;
    esac
done <<< "$all_changed"

if [ ${#affected[@]} -eq 0 ]; then
    echo ":: Unknown changes (no pattern matched), running all C++ tests"
    ctest --test-dir build -E wrappers -LE "(MPI|NotImplemented)" \
        --output-on-failure
    exit
fi

# Build an anchored ctest regex.  The (_[0-9]+)? suffix accounts for
# multi-argument tests that get a numeric suffix (e.g. tests_core_tile_gemm_1).
patterns=$(printf '%s\n' "${!affected[@]}" | sort | paste -sd '|')
regex="^(${patterns})(_[0-9]+)?$"

echo ":: Running ${#affected[@]} affected C++ test pattern(s):"
printf '  - %s\n' "${!affected[@]}" | sort
echo ":: CTest regex: $regex"

ctest --test-dir build -R "$regex" -E wrappers -LE "(MPI|NotImplemented)" \
    --output-on-failure
