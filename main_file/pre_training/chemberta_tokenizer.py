from transformers import RobertaTokenizerFast
from datasets import load_dataset, Dataset, DatasetDict
import os

# charging of tokenizer ChemBERTa
tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Charging data_output_pretraining
dataset = load_dataset("text", data_files={"train": "data_output_pretraining/filtred_ligands_raw.txt"})["train"]

# Split 80% train, 10% val, 10% test
dataset_split = dataset.train_test_split(test_size=0.2)  # 80% train, 20% test
train_dataset = dataset_split['train']  # Training Dataset (80%)
test_val_dataset = dataset_split['test']   # Dataset test + validation (20%)

# Split in two part the 20% for test/validation
test_val_split = test_val_dataset.train_test_split(test_size=0.5)  # 50% test, 50% validation
val_dataset = test_val_split['train']  # Dataset de validation (10%)
test_dataset = test_val_split['test']  # Dataset de test (10%)

# Check the dataset size
print(f"training: {len(train_dataset)} examples")
print(f"Validation: {len(val_dataset)} examples")
print(f"Test: {len(test_dataset)} examples")

# Tokenization function with padding + truncation
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=90)

# Tokenization of each set
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Save each tokenized set
tokenized_train_dataset.save_to_disk("data_output_pretraining/tok_lig_pretraining3/train")
tokenized_val_dataset.save_to_disk("data_output_pretraining/tok_lig_pretraining3/val")
tokenized_test_dataset.save_to_disk("data_output_pretraining/tok_lig_pretraining3/test")

print("Dataset tokenized and saved.")
