# Orchestration Makefile for rebuilding StarPU + graph scheduler, NNTile
# (nntile_core + nntile_graph), and running the GPT-2 LMHead notebook workflow with small
# sequence/batch settings.
#
# Override paths if your checkout layout differs:
#   make build STARPU_SRC=/path/to/starpu-sched
#
# Optional: STARPU_BUILD (default: $(STARPU_SRC)/build), NNTILE_BUILD,
# GRAPH_SCHED_DIR (default: $(STARPU_SRC)/new_sched), CMAKE.

THIS_DIR        := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
NNTILE_ROOT     := $(abspath $(THIS_DIR)/..)
NNTILE_BUILD    ?= $(NNTILE_ROOT)/build

STARPU_SRC      ?= $(abspath $(NNTILE_ROOT)/../starpu-sched)
STARPU_BUILD    ?= $(STARPU_SRC)/build
GRAPH_SCHED_DIR ?= $(STARPU_SRC)/new_sched

CMAKE           ?= cmake
GRAPH_SCHED_SO  := $(abspath $(GRAPH_SCHED_DIR)/libgraph_recorder_sched.so)
GRAPH_SCHED_LD  := $(abspath $(GRAPH_SCHED_DIR))

NOTEBOOK_DIR    := $(NNTILE_ROOT)/notebooks
NNTILE_PYTHON   := $(NNTILE_BUILD)/wrappers/python
TRAIN_SEED      ?= 0

# Env vars from notebooks/gpt2_lmhead.ipynb (single line for one shell)
STARPU_ENV := CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
	PYTHONPATH=$(NNTILE_PYTHON) \
	STARPU_NCPU=1 STARPU_NCUDA=0 STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1 \
	STARPU_PROFILING=1 STARPU_BUS_STATS=1 \
	STARPU_HOME=$(NOTEBOOK_DIR)/starpu \
	STARPU_PERF_MODEL_DIR=$(NOTEBOOK_DIR)/starpu/sampling \
	STARPU_PERF_MODEL_HOMOGENEOUS_CPU=1 STARPU_PERF_MODEL_HOMOGENEOUS_CUDA=1 \
	STARPU_HOSTNAME=GPT2_LMHead_example \
	STARPU_FXT_PREFIX=$(NOTEBOOK_DIR)/starpu/fxt \
	STARPU_GRAPH_SCHED_VERBOSE=5 \
	STARPU_GRAPH_SCHED_CHECKPOINT_MAX=1000

# Training invocation from notebook §3.1, with requested hyperparameters
GPT2_TRAIN := python ../wrappers/python/examples/gpt2_lmhead_training.py \
	--restrict=cuda \
	--pretrained=local \
	--config-path=../wrappers/python/examples/gpt2_default_config.json \
	--save-checkpoint-path=.model/nntile_checkpoint.pt \
	--optimizer=adam \
	--lr=1e-4 \
	--dtype=fp32 \
	--nepochs=5 \
	--seq-len=32 \
	--batch-size=4 \
	--minibatch-size=4 \
	--dataset-file=tinystories/train.bin \
	--seed=$(TRAIN_SEED)

.PHONY: build build_starpu install_starpu build_graph_sched configure_nntile \
	nntile_python nntile_core nntile_graph \
	data run run_dmdasd run_graph run_grpah

# ---------------------------------------------------------------------------
# build: StarPU (compile + install) -> graph scheduler (in-tree, no install)
#        -> CMake configure NNTile (incremental) -> nntile_core + nntile_graph
# ---------------------------------------------------------------------------
build: build_starpu install_starpu build_graph_sched configure_nntile nntile_python

build_starpu:
	$(MAKE) -C $(STARPU_BUILD)

install_starpu:
	$(MAKE) -C $(STARPU_BUILD) install

build_graph_sched:
	$(MAKE) -C $(GRAPH_SCHED_DIR)

configure_nntile:
	$(CMAKE) -S $(NNTILE_ROOT) -B $(NNTILE_BUILD)

# Pybind extensions into $(NNTILE_PYTHON)/nntile/ (core ops + graph API)
nntile_python: configure_nntile
	$(CMAKE) --build $(NNTILE_BUILD) --target nntile_core --target nntile_graph --parallel

nntile_core: configure_nntile
	$(CMAKE) --build $(NNTILE_BUILD) --target nntile_core --parallel

nntile_graph: configure_nntile
	$(CMAKE) --build $(NNTILE_BUILD) --target nntile_graph --parallel

# ---------------------------------------------------------------------------
# data: causal_lm_data_preparation.py (same script as gpt2_lmhead.ipynb)
# ---------------------------------------------------------------------------
data:
	cd $(NOTEBOOK_DIR) && python ../wrappers/python/examples/causal_lm_data_preparation.py \
		--hf-tokenizer=openai-community/gpt2 \
		--seq-len=32 \
		--batch-size=4 \
		--dataset-select=1

# PYTHONPATH -> $(NNTILE_BUILD)/wrappers/python (via STARPU_ENV); default: graph scheduler
run: run_graph

run_dmdasd:
	cd $(NOTEBOOK_DIR) && $(STARPU_ENV) STARPU_SCHED=dmdasd $(GPT2_TRAIN)

# Loadable graph_recorder policy from $(GRAPH_SCHED_DIR) (see that tree's Makefile)
run_graph:
	cd $(NOTEBOOK_DIR) && $(STARPU_ENV) STARPU_SCHED=graph_recorder \
		STARPU_SCHED_LIB=$(GRAPH_SCHED_SO) \
		LD_LIBRARY_PATH=$(GRAPH_SCHED_LD):$$LD_LIBRARY_PATH \
		$(GPT2_TRAIN)

run_grpah: run_graph
