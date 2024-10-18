import datasets
import torch
from transformers import (
    AdamW, BertConfig, BertForMaskedLM, BertTokenizerFast,
    DataCollatorForLanguageModeling)

dataset = datasets.load_dataset('oscar', 'unshuffled_deduplicated_la',
                                split="train",
                                cache_dir=".data")


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased",
                                              cache_dir=".model")


def batch_iterator(batch_size=10000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


bert_tokenizer_trained = tokenizer.train_new_from_iterator(
    text_iterator=batch_iterator(),
    vocab_size=32_000,
    max_len=512)


class TokenizedDataset(torch.utils.data.Dataset):
    "This wraps the dataset and tokenizes it, ready for the model"

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.tokenizer.encode(
            self.dataset[i]["text"],
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )[0, ...]


tokenized_dataset = TokenizedDataset(dataset, bert_tokenizer_trained)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=bert_tokenizer_trained,
    mlm=True,
    mlm_probability=0.2,
    return_tensors="pt",
)

train_loader = torch.utils.data.DataLoader(tokenized_dataset,
                                           batch_size=32,
                                           collate_fn=data_collator)

config = BertConfig(
    vocab_size=32000,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
    )

model = BertForMaskedLM(config)

device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')
# and move our model over to the selected device
model.to(device)
print(device)

# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=1e-4)

epochs = 3

for epoch in range(epochs):
    print("Epoch", epoch)
    # setup loop with TQDM and dataloader
    for batch in train_loader:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        # attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        print(labels)
        # process
        outputs = model(input_ids,
                        # attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        print("loss", loss.item())
        break
