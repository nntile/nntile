import numpy as np
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


class MemmapBatchedReaderDataset(IterableDataset):
    def __init__(self, memmap_array, batch_size, seq_size):
        self.memmap_array = memmap_array
        self.batch_size = batch_size
        self.seq_size = seq_size

        self.batch_elems_size = self.seq_size * self.batch_size

        assert len(self.memmap_array) % (self.batch_elems_size) == 0
        self.num_batches = len(self.memmap_array) // (self.batch_elems_size)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, batch_ind):
        mem_begin = self.batch_elems_size * batch_ind
        mem_end = mem_begin + self.batch_elems_size
        batch_data = self.memmap_array[mem_begin:mem_end]
        batch_data = batch_data.reshape((self.batch_size, self.seq_size))
        return batch_data

    def __iter__(self):
        for batch_ind in range(len(self)):
            yield self[batch_ind]


class RoBERTaBatchedMaskingWrapper(IterableDataset):
    """
    Create random mask on each entry. Mask is uqique on each __getitem__ call
    """

    def __init__(self, dataset, tokenizer, mask_amount, num_masks=1):
        self.dataset = dataset
        self.num_masks = num_masks

        self.mask_elems = (
            mask_amount
            if isinstance(mask_amount, int)
            else int(len(dataset) * mask_amount)
        )

        assert self.mask_elems > 0

        (self.cls_token, self.mask_token, self.pad_token, self.sep_token) = (
            tokenizer.encode("[MASK] [PAD]")
        )

    def __len__(self):
        return len(self.dataset)

    def get_mask(self, batch):
        rng = np.random.default_rng()

        mask_index = np.zeros_like(batch, dtype=bool)
        for batch_entry in range(batch.shape[0]):
            indexes = rng.choice(
                np.arange(batch.shape[1]), self.mask_elems, replace=False
            )
            mask_index[batch_entry, indexes] = True

        return mask_index

    def __getitem__(self, elem_ind):
        batch = self.dataset[elem_ind]
        masked = np.array(batch)

        mask_index = self.get_mask(batch)
        masked[mask_index] = self.mask_token

        return batch, masked

    def __iter__(self):
        for batch_ind in range(len(self)):
            batch, mask = self[batch_ind]
            yield batch, mask


class BertBatchedMaskingWrapper(IterableDataset):
    """
    Create random mask on each entry. For each entry there are several masks.
    But masks dont changed between calls and so dataset is statically known
    """

    def __init__(self, dataset, tokenizer, mask_amount, num_masks=1):
        self.dataset = dataset
        self.num_masks = num_masks

        self.mask_elems = (
            mask_amount
            if isinstance(mask_amount, int)
            else int(len(dataset) * mask_amount)
        )
        self.masks_storage = self._fill_mask_storage()

        assert self.mask_elems > 0

        (self.cls_token, self.mask_token, self.pad_token, self.sep_token) = (
            tokenizer.encode("[MASK] [PAD]")
        )

    def __len__(self):
        return len(self.dataset) * self.num_masks

    def __getitem__(self, elem_ind):
        batch_ind = elem_ind // self.num_masks
        mask_ind = elem_ind % self.num_masks

        batch = self.dataset[batch_ind]
        masked = np.array(batch)

        mask_index = self.masks_storage[batch_ind][mask_ind]
        masked[mask_index] = self.mask_token

        return batch, masked

    def __iter__(self):
        for batch_ind in range(len(self)):
            batch, mask = self[batch_ind]
            yield batch, mask

    def _fill_mask_storage(self):
        rng = np.random.default_rng()

        masks = []
        for batch_id in range(len(self.dataset)):
            data_x = self.dataset[batch_id]
            batch_masks = []
            for _ in range(self.num_masks):
                mask = np.zeros_like(data_x, dtype=bool)
                for batch_entry in range(data_x.shape[0]):
                    indexes = rng.choice(
                        np.arange(data_x.shape[1]),
                        self.mask_elems,
                        replace=False,
                    )
                    mask[batch_entry, indexes] = True

                batch_masks.append(mask)

            masks.append(batch_masks)

        return masks


if __name__ == "__main__":
    a = np.memmap(".data/tinystories/train.bin", mode="r")
    a = a[:100]

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", cache_dir=".data"
    )

    batch_size = 2
    seq_size = 10

    d = MemmapBatchedReaderDataset(a, batch_size, seq_size)

    dw = BertBatchedMaskingWrapper(d, tokenizer, 1, 2)
    drw = RoBERTaBatchedMaskingWrapper(d, tokenizer, 1, 2)

    for x_batch, y_batch in dw:
        pass

    for x_batch, y_batch in drw:
        pass
