"""Data loading, tokenization, dynamic padding, and metadata extraction pipeline."""

from typing import Dict, Any, Tuple, List
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_and_preprocess_glue_dataset(
    dataset_name: str,
    tokenizer_name_or_path: str,
    max_seq_length: int = 128
) -> Tuple[Any, PreTrainedTokenizerBase, Dict[str, Any], List[str]]:
    """Load GLUE task dataset, tokenize splits, extract raw text, and gather metadata.

    Args:
        dataset_name: One of 'sst2', 'mrpc', 'rte'.
        tokenizer_name_or_path: Hugging Face model identifier (e.g. 'bert-base-uncased').
        max_seq_length: Maximum sequence length for truncation.

    Returns:
        Tuple of (tokenized_datasets, tokenizer, dataset_metadata, raw_val_texts).
    """
    try:
        raw_dataset = load_dataset("nyu-mll/glue", dataset_name)
    except Exception:
        raw_dataset = load_dataset("glue", dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)

    # Extract dataset provenance metadata for run.json
    train_split = raw_dataset["train"]
    dataset_meta = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_name,
        "huggingface_builder": getattr(raw_dataset, "builder_name", "nyu-mll/glue"),
        "dataset_fingerprint": str(getattr(train_split, "_fingerprint", "N/A")),
        "num_train_samples": len(raw_dataset["train"]),
        "num_validation_samples": len(raw_dataset["validation"]),
        "num_labels": len(raw_dataset["train"].features["label"].names),
        "label_names": raw_dataset["train"].features["label"].names
    }

    # Extract raw text for misclassified failure analysis
    val_split = raw_dataset["validation"]
    raw_val_texts: List[str] = []
    for item in val_split:
        if dataset_name == "sst2":
            raw_val_texts.append(str(item["sentence"]))
        else:  # mrpc or rte
            raw_val_texts.append(f"{item['sentence1']} | {item['sentence2']}")

    # Tokenization preprocessing function
    def preprocess_function(examples):
        if dataset_name == "sst2":
            args = (examples["sentence"],)
        else:
            args = (examples["sentence1"], examples["sentence2"])
        
        return tokenizer(
            *args,
            truncation=True,
            max_length=max_seq_length,
            padding=False  # Dynamic padding during batch collation via DataCollator
        )

    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Tokenizing {dataset_name} dataset"
    )

    return tokenized_dataset, tokenizer, dataset_meta, raw_val_texts
