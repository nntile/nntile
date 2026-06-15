# BERT graph model tests

Reference data is generated from HuggingFace `modeling_bert` via
`generate_test_data.py` (PyTorch eager, graph tensors). All block
forwards and backwards call HF modules directly; layout helpers only reshape
weights for the graph API. Run `python nntile/tests/model/bert/test_bert_generate_hf_parity.py`
to verify HF parity locally.

Layout: hidden states `(batch, seq, hidden_size)`; `input_ids` /
`token_type_ids` / `position_ids` are `(batch, seq)`.

`BertMlm` outputs logits `(batch, seq, vocab_size)`.

Blocks: `intermediate`, `attention`, `layer`, `embeddings`, `model`, `mlm`.
