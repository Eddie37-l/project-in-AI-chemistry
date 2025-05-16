from transformers import RobertaTokenizerFast
from datasets import load_dataset, Dataset, DatasetDict
import os

# Chargement du tokenizer ChemBERTa
tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Charger les données brutes (par exemple à partir d'un fichier .txt)
dataset = load_dataset("text", data_files={"train": "data/ligands_raw.txt"})["train"]

# Séparation en 80% train, 10% val, 10% test
dataset_split = dataset.train_test_split(test_size=0.2)  # 80% train, 20% test
train_dataset = dataset_split['train']  # Dataset d'entraînement (80%)
test_val_dataset = dataset_split['test']   # Dataset test + validation (20%)

# Séparer encore en 90% train, 10% validation pour la partie test/validation
test_val_split = test_val_dataset.train_test_split(test_size=0.5)  # 50% test, 50% validation (donc 10% du total pour la validation)
val_dataset = test_val_split['train']  # Dataset de validation (10%)
test_dataset = test_val_split['test']  # Dataset de test (10%)

# Afficher les tailles des datasets pour vérifier
print(f"Entraînement: {len(train_dataset)} exemples")
print(f"Validation: {len(val_dataset)} exemples")
print(f"Test: {len(test_dataset)} exemples")

# Fonction de tokenisation avec padding + truncation
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Appliquer la tokenisation sur chaque sous-ensemble
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Sauvegarder les datasets tokenisés
tokenized_train_dataset.save_to_disk("data/tokenized_ligands/train")
tokenized_val_dataset.save_to_disk("data/tokenized_ligands/val")
tokenized_test_dataset.save_to_disk("data/tokenized_ligands/test")

print("Dataset tokenisé sauvegardé.")
