import torch
from torch.utils.data import Dataset, DataLoader, random_split

import tiktoken

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        # shift the token ids to the left by one
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_data_loader(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


def create_data_loaders(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle_train=True,
    drop_last=True,
    num_workers=0,
    train_val_test_split=(0.8, 0.1, 0.1),
    seed=42,
):
    """
    Create train/val/test DataLoaders from raw text using a token chunking dataset.

    Args:
        text: Full corpus text
        batch_size: Batch size for all splits
        max_length: Sequence length for model inputs
        stride: Stride for sliding window tokenization
        shuffle_train: Whether to shuffle only the training loader
        drop_last: Drop last incomplete batch (applied to all loaders)
        num_workers: DataLoader workers
        train_val_test_split: Tuple of ratios summing to 1.0
        seed: RNG seed for deterministic splitting

    Returns:
        (train_loader, val_loader, test_loader)
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)

    total_len = len(dataset)
    train_ratio, val_ratio, test_ratio = train_val_test_split
    # Convert ratios to lengths ensuring they sum to total_len
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = max(0, total_len - train_len - val_len)

    # Ensure at least one sample in train if possible
    if train_len == 0 and total_len > 0:
        train_len = 1
        if val_len > 0:
            val_len -= 1
        else:
            test_len = max(0, test_len - 1)

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader