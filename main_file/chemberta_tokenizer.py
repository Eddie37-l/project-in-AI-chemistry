# The goal is to tokenize the desired database and prepare it for the masking
from transformers import RobertaTokenizerFast
from datasets import load_dataset, Dataset
import os

# Official tokenizer ofChemBERTad
tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Charger les données brutes
dataset = load_dataset("text", data_files={"train": "data/ligands_raw.txt"})["train"]

# Tokenizer avec padding + truncation
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Apply the tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Save the tokenized dataset
tokenized_dataset.save_to_disk("data/tokenized_ligands")

print("Dataset tokenisé sauvegardé.")