"""
AG News dataset loading and preprocessing.

Uses HuggingFace datasets + tokenizers (no torchtext dependency).
Supports offline: if Hub is unreachable, loads from data_root (see doc below).
"""

import logging
import os
import re
from typing import Tuple, Optional, List, Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

# Optional: HuggingFace datasets + tokenizers
try:
    from datasets import load_dataset, load_from_disk
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from tokenizers.normalizers import Lowercase
    HF_AVAILABLE = True
except ImportError as e:
    HF_AVAILABLE = False
    load_dataset = None
    Tokenizer = None
    logger.warning(
        "HuggingFace 'datasets' or 'tokenizers' not available. "
        "AG News requires: pip install datasets tokenizers. Error: %s",
        e,
    )


# Special token IDs (must match tokenizer special tokens order)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN]


def _simple_preprocess(text: str) -> str:
    """Lowercase and basic cleanup, similar to basic_english."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _build_tokenizer_from_iterator(
    text_iterator,
    vocab_size: Optional[int] = 50000,
    min_frequency: int = 2,
) -> "Tokenizer":
    """Train a WordLevel tokenizer from an iterator of texts."""
    tokenizer = Tokenizer(models.WordLevel(unk_token=UNK_TOKEN))
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size or 50000,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(text_iterator, trainer)
    return tokenizer


class VocabWrapper:
    """Thin wrapper so 'vocab' has len(vocab) and tokenizer for encoding (compatible with existing code)."""

    def __init__(self, tokenizer: "Tokenizer"):
        self._tokenizer = tokenizer
        self._pad_id = tokenizer.token_to_id(PAD_TOKEN)
        self._unk_id = tokenizer.token_to_id(UNK_TOKEN)
        if self._pad_id is None:
            self._pad_id = 0
        if self._unk_id is None:
            self._unk_id = 0

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode_tokens(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Tokenize text and return list of token ids."""
        enc = self._tokenizer.encode(_simple_preprocess(text))
        ids = enc.ids
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def __getitem__(self, key: str) -> int:
        """Allow vocab['<pad>'] / vocab['<unk>'] for compatibility."""
        tid = self._tokenizer.token_to_id(key)
        return tid if tid is not None else self._unk_id


class AGNewsDataset(Dataset):
    """PyTorch Dataset over HuggingFace ag_news (label, text) for DataLoader compatibility."""

    def __init__(self, hf_split, label_key: str = "label", text_key: str = "text"):
        self._data = hf_split
        self._label_key = label_key
        self._text_key = text_key

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[int, str]:
        row = self._data[idx]
        return (int(row[self._label_key]), str(row[self._text_key]))


def _collate_batch(
    batch: List[Tuple[int, str]],
    vocab: VocabWrapper,
    max_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate (label, text) batches into (labels, texts_tensor, lengths)."""
    labels_list, texts_list = zip(*batch)

    text_indices = [vocab.encode_tokens(t, max_length=max_length) for t in texts_list]
    lengths = torch.tensor([len(ids) for ids in text_indices], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) > 0 else 1
    pad_id = vocab.pad_id

    padded = []
    for ids in text_indices:
        padded.append(ids + [pad_id] * (max_len - len(ids)))

    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    texts_tensor = torch.tensor(padded, dtype=torch.long)
    return labels_tensor, texts_tensor, lengths


def _create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    collate_fn,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn,
    )


def get_agnews_dataloaders(
    data_root: str = "./datasets/agnews",
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: Optional[int] = None,
    max_length: Optional[int] = None,
    vocab_size: Optional[int] = 50000,
    min_frequency: int = 2,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, Any]:
    """
    Get AG News dataloaders using HuggingFace datasets + tokenizers.

    Args:
        data_root: Ignored (data from HuggingFace Hub); kept for API compatibility.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        val_split: Fraction of training data used for validation.
        seed: Random seed for train/val split.
        max_length: Max sequence length (None = no truncation).
        vocab_size: Max vocabulary size for WordLevel tokenizer.
        min_frequency: Min token frequency for tokenizer training.

    Returns:
        (train_loader, val_loader, test_loader, vocab).
        vocab has len(vocab) and is used for model vocab_size.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "AG News requires HuggingFace 'datasets' and 'tokenizers'. "
            "Install with: pip install datasets tokenizers"
        )

    # Load AG News: try Hub first, then local data_root (offline)
    dataset = None
    try:
        dataset = load_dataset("ag_news")
    except (ConnectionError, OSError) as e:
        if os.path.isdir(data_root):
            try:
                dataset = load_from_disk(data_root)
                logger.info("Loaded AG News from local path: %s", data_root)
            except Exception as local_e:
                logger.warning("Local load failed: %s", local_e)
        if dataset is None:
            raise RuntimeError(
                "Could not load AG News: Hub unreachable and no local copy at %r. "
                "Either run with internet once (data will be cached), or save offline:\n"
                "  from datasets import load_dataset; d = load_dataset('ag_news'); d.save_to_disk(%r)"
                % (data_root, data_root)
            ) from e

    train_hf = dataset["train"]
    test_hf = dataset["test"]

    # Train tokenizer on training text
    def train_text_iterator():
        for i in range(len(train_hf)):
            yield _simple_preprocess(train_hf[i]["text"])

    tokenizer = _build_tokenizer_from_iterator(
        train_text_iterator(),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    vocab = VocabWrapper(tokenizer)
    logger.info("AG News vocabulary size: %d", len(vocab))

    def collate_fn(batch):
        return _collate_batch(batch, vocab, max_length=max_length)

    train_dataset = AGNewsDataset(train_hf)
    test_dataset = AGNewsDataset(test_hf)

    if val_split > 0 and val_split < 1:
        n = len(train_dataset)
        val_size = int(n * val_split)
        train_size = n - val_size
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size], generator=generator
        )
    else:
        train_subset = train_dataset
        val_subset = None

    train_loader = _create_optimized_dataloader(
        train_subset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )
    val_loader = (
        _create_optimized_dataloader(
            val_subset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
        )
        if val_subset is not None
        else None
    )
    test_loader = _create_optimized_dataloader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )

    n_train = len(train_subset)
    n_val = len(val_subset) if val_subset is not None else 0
    n_test = len(test_dataset)
    logger.info(
        "AG News dataloaders: train=%d, val=%d, test=%d",
        n_train, n_val, n_test,
    )

    return train_loader, val_loader, test_loader, vocab
