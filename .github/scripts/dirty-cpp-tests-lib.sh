#!/usr/bin/env bash
# Shared logic: map a git diff to CTest names / CMake targets for dirty C++ tests.
# Sourced by run-dirty-cpp-tests.sh and ci-dirty-cpp-tests.sh.
#
# @file .github/scripts/dirty-cpp-tests-lib.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "source dirty-cpp-tests-lib.sh, do not execute" >&2
    exit 1
fi

NNTILE_DIRTY_RUN_ALL=false
declare -A NNTILE_DIRTY_AFFECTED=()

nntile_dirty_cpp_reset_plan() {
    NNTILE_DIRTY_RUN_ALL=false
    NNTILE_DIRTY_AFFECTED=()
}

# ---------- helper functions for layer propagation -------------------------
add_all_layers() {
    local op=$1
    for p in tests_kernel tests_starpu tests_core \
             tests_graph_tile_ops tests_graph_tensor_ops; do
        NNTILE_DIRTY_AFFECTED["${p}_${op}"]=1
    done
}

add_from_starpu() {
    local op=$1
    for p in tests_starpu tests_core tests_graph_tile_ops \
             tests_graph_tensor_ops; do
        NNTILE_DIRTY_AFFECTED["${p}_${op}"]=1
    done
}

add_from_tile() {
    local op=$1
    for p in tests_core tests_graph_tile_ops tests_graph_tensor_ops; do
        NNTILE_DIRTY_AFFECTED["${p}_${op}"]=1
    done
}

# GPT-Neo graph model: run all block tests when shared code changes.
add_gptneo_model_tests() {
    for t in gptneo_config gptneo_mlp gptneo_attention gptneo_decoder \
             gptneo_model gptneo_causal; do
        NNTILE_DIRTY_AFFECTED["tests_graph_model_${t}"]=1
    done
}

add_t5_model_tests() {
    for t in t5_config t5_ff t5_attention t5_cross_attention \
             t5_encoder_block t5_decoder_block t5_model t5_conditional; do
        NNTILE_DIRTY_AFFECTED["tests_graph_model_${t}"]=1
    done
}

add_bert_model_tests() {
    NNTILE_DIRTY_AFFECTED["tests_graph_model_bert_config"]=1
    for t in bert_intermediate bert_attention bert_layer \
             bert_embeddings bert_model bert_mlm; do
        NNTILE_DIRTY_AFFECTED["tests_graph_model_${t}"]=1
        NNTILE_DIRTY_AFFECTED["tests_graph_model_${t}_data_setup"]=1
    done
}

# RoBERTa graph tests import safetensor layout helpers from
# nntile/tests/model/bert/generate_test_data.py only (not other BERT sources).
add_roberta_model_tests() {
    NNTILE_DIRTY_AFFECTED["tests_graph_model_roberta_config"]=1
    for t in roberta_intermediate roberta_attention roberta_layer \
             roberta_embeddings roberta_model roberta_mlm; do
        NNTILE_DIRTY_AFFECTED["tests_graph_model_${t}"]=1
        NNTILE_DIRTY_AFFECTED["tests_graph_model_${t}_data_setup"]=1
    done
}

# Returns 0 when the diff is empty.
nntile_dirty_cpp_collect() {
    local base_rev=$1
    local head_rev=${2:-HEAD}
    nntile_dirty_cpp_reset_plan
    echo ":: Diff ${base_rev}..${head_rev}"
    local all_changed
    all_changed=$(git diff --name-only "${base_rev}..${head_rev}" || true)
    if [ -z "$all_changed" ]; then
        echo ":: No files changed"
        return 0
    fi
    local run_all=false
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
            nntile/tests/model/bert/generate_test_data.py)
                run_all=true; break ;;
        esac
    done <<< "$all_changed"
    if $run_all; then
        NNTILE_DIRTY_RUN_ALL=true
        echo ":: Core files changed, full C++ test suite is dirty"
        return 0
    fi

    while IFS= read -r file; do
        [ -z "$file" ] && continue

        case "$file" in
        # ---- test files: run the specific test ----------------------------
        nntile/tests/constants.cc)
            NNTILE_DIRTY_AFFECTED["tests_nntile_constants"]=1 ;;
        nntile/tests/kernel/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_kernel_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/starpu/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_starpu_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/core/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_core_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/tensor/ops/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_tensor_ops_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/tensor/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_tensor_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/tile/ops/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_tile_ops_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/tile/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_tile_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/nn/ops/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_nn_ops_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/nn/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_nn_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/module/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_module_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/io/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_io_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/llama/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/gpt2/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/gptneo/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/t5/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/roberta/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        nntile/tests/model/test_gptneo_fixture_helpers.hh)
            add_gptneo_model_tests ;;
        nntile/tests/model/test_t5_fixture_helpers.hh)
            add_t5_model_tests ;;
        nntile/tests/model/bert/generate_test_data.py)
            add_bert_model_tests
            add_roberta_model_tests ;;
        nntile/tests/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_$(basename "$file" .cc)"]=1 ;;

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

        # ---- graph-level: only the matching test --------------------------
        # ops/ before parent dir: in case, * matches / so tensor/*.cc also
        # matches tensor/ops/foo.cc unless ops/ or [^/]* is listed first.
        nntile/src/tensor/ops/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_tensor_ops_$(basename "$file" .cc)"]=1 ;;
        include/nntile/tensor/ops/*.hh)
            NNTILE_DIRTY_AFFECTED["tests_graph_tensor_ops_$(basename "$file" .hh)"]=1 ;;
        nntile/src/tensor/[^/]*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_tensor_$(basename "$file" .cc)"]=1 ;;
        include/nntile/tensor/[^/]*.hh)
            NNTILE_DIRTY_AFFECTED["tests_graph_tensor_$(basename "$file" .hh)"]=1 ;;
        nntile/src/tile/ops/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_tile_ops_$(basename "$file" .cc)"]=1 ;;
        include/nntile/tile/ops/*.hh)
            NNTILE_DIRTY_AFFECTED["tests_graph_tile_ops_$(basename "$file" .hh)"]=1 ;;
        nntile/src/tile/[^/]*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_tile_$(basename "$file" .cc)"]=1 ;;
        include/nntile/tile/[^/]*.hh)
            NNTILE_DIRTY_AFFECTED["tests_graph_tile_$(basename "$file" .hh)"]=1 ;;
        nntile/src/nn/ops/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_nn_ops_$(basename "$file" .cc)"]=1 ;;
        include/nntile/nn/ops/*.hh)
            NNTILE_DIRTY_AFFECTED["tests_graph_nn_ops_$(basename "$file" .hh)"]=1 ;;
        nntile/src/nn/[^/]*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_nn_$(basename "$file" .cc)"]=1 ;;
        include/nntile/nn/[^/]*.hh)
            NNTILE_DIRTY_AFFECTED["tests_graph_nn_$(basename "$file" .hh)"]=1 ;;
        nntile/src/module/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_module_$(basename "$file" .cc)"]=1 ;;
        include/nntile/module/*.hh)
            NNTILE_DIRTY_AFFECTED["tests_graph_module_$(basename "$file" .hh)"]=1 ;;
        nntile/src/io/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_io_$(basename "$file" .cc)"]=1 ;;
        include/nntile/io/*.hh)
            NNTILE_DIRTY_AFFECTED["tests_graph_io_$(basename "$file" .hh)"]=1 ;;
        nntile/src/model/llama/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        include/nntile/model/llama/*.hh)
            NNTILE_DIRTY_AFFECTED["tests_graph_model_$(basename "$file" .hh)"]=1 ;;
        nntile/src/model/gpt2/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
        include/nntile/model/gpt2/*.hh)
            NNTILE_DIRTY_AFFECTED["tests_graph_model_$(basename "$file" .hh)"]=1 ;;
        nntile/src/model/gptneo/*.cc)
            NNTILE_DIRTY_AFFECTED["tests_graph_model_$(basename "$file" .cc)"]=1 ;;
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

    if [ ${#NNTILE_DIRTY_AFFECTED[@]} -eq 0 ]; then
        NNTILE_DIRTY_RUN_ALL=true
        echo ":: Unknown changes (no pattern matched), full C++ test suite is dirty"
        return 0
    fi
    echo ":: ${#NNTILE_DIRTY_AFFECTED[@]} dirty CTest name(s)"
    return 0
}

# Optional filter for local core|graph split jobs.
nntile_dirty_cpp_filter_label() {
    local ctest_label=$1
    [ -n "$ctest_label" ] || return 0
    case "$ctest_label" in
        core|graph) ;;
        *) echo "Unknown ctest label filter: ${ctest_label}" >&2; return 2 ;;
    esac
    declare -A layer_affected=()
    local name
    for name in "${!NNTILE_DIRTY_AFFECTED[@]}"; do
        case "$name" in
            tests_"${ctest_label}"_*)
                layer_affected["$name"]=1
                ;;
        esac
    done
    NNTILE_DIRTY_AFFECTED=()
    for name in "${!layer_affected[@]}"; do
        NNTILE_DIRTY_AFFECTED["$name"]=1
    done
    if [ ${#NNTILE_DIRTY_AFFECTED[@]} -eq 0 ]; then
        echo ":: No ${ctest_label}-layer tests dirty; skip"
        return 1
    fi
    return 0
}

nntile_dirty_cpp_ctest_regex() {
    if $NNTILE_DIRTY_RUN_ALL; then
        echo ""
        return 0
    fi
    local patterns
    patterns=$(printf '%s\n' "${!NNTILE_DIRTY_AFFECTED[@]}" | sort | paste -sd '|')
    echo "^(${patterns})(_[0-9]+)?$"
}

# CMake executable targets (exclude CTest-only fixture setup steps).
nntile_dirty_cpp_cmake_targets() {
    if $NNTILE_DIRTY_RUN_ALL; then
        echo ""
        return 0
    fi
    local name
    for name in $(printf '%s\n' "${!NNTILE_DIRTY_AFFECTED[@]}" | sort); do
        case "$name" in
            *_data_setup) continue ;;
        esac
        printf '%s ' "$name"
    done
}

nntile_dirty_cpp_emit_plan() {
    if $NNTILE_DIRTY_RUN_ALL; then
        echo "export NNTILE_DIRTY_RUN_ALL=1"
        echo "export NNTILE_DIRTY_SKIP=0"
        echo "export NNTILE_DIRTY_CMAKE_TARGETS="
        echo "export NNTILE_DIRTY_CTEST_REGEX="
        return 0
    fi
    if [ ${#NNTILE_DIRTY_AFFECTED[@]} -eq 0 ]; then
        echo "export NNTILE_DIRTY_RUN_ALL=0"
        echo "export NNTILE_DIRTY_SKIP=1"
        echo "export NNTILE_DIRTY_CMAKE_TARGETS="
        echo "export NNTILE_DIRTY_CTEST_REGEX="
        return 0
    fi
    local regex targets
    regex=$(nntile_dirty_cpp_ctest_regex)
    targets=$(nntile_dirty_cpp_cmake_targets)
    echo "export NNTILE_DIRTY_RUN_ALL=0"
    echo "export NNTILE_DIRTY_SKIP=0"
    printf 'export NNTILE_DIRTY_CMAKE_TARGETS=%q\n' "$targets"
    printf 'export NNTILE_DIRTY_CTEST_REGEX=%q\n' "$regex"
}
