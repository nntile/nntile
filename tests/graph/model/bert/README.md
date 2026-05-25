# BERT graph model tests

Reference data is generated from HuggingFace `modeling_bert` via
`generate_test_data.py` (PyTorch eager, Fortran-order tensors).

Layout: hidden states `(hidden_size, seq, batch)`; `input_ids` /
`token_type_ids` / `position_ids` are `(seq, batch)`.

`BertMlm` outputs logits `(vocab_size, seq, batch)`.

Blocks: `intermediate`, `attention`, `layer`, `embeddings`, `model`, `mlm`.
