import argparse
import os
import wandb
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_from_disk, DatasetDict

# ----------------------------
# Argument parsing
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="Répertoire contenant les datasets 'train' et 'val'")
parser.add_argument("--output_dir", type=str, default="output/chemberta-mlm", help="Répertoire de sortie des résultats")
parser.add_argument("--save_dir", type=str, default="models/chemberta-mlm-custom", help="Répertoire pour sauvegarder le modèle")
parser.add_argument("--batch_size", type=int, default=16, help="Taille du batch")
parser.add_argument("--epochs", type=int, default=5, help="Nombre d’époques")
parser.add_argument("--save_steps", type=int, default=500, help="Fréquence de sauvegarde du modèle")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Taux d'apprentissage")
args = parser.parse_args()

# ----------------------------
# Init WandB
# ----------------------------
wandb.init(project="chemberta-pretraining", name="ChemBERTa MLM Custom")

# ----------------------------
# Load tokenizer and model
# ----------------------------
print("🔹 Chargement du tokenizer...")
tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
print("✅ Tokenizer chargé.")

print("🔹 Chargement du modèle...")
model = RobertaForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
print("✅ Modèle chargé.")

# ----------------------------
# Load dataset
# ----------------------------
print(f"🔹 Chargement du dataset depuis {args.dataset_dir}...")
train_dataset = load_from_disk(os.path.join(args.dataset_dir, "train"))
val_dataset = load_from_disk(os.path.join(args.dataset_dir, "val"))

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})
print(f"✅ Dataset chargé. Train : {len(dataset['train'])}, Validation : {len(dataset['validation'])}")

# ----------------------------
# Data collator
# ----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# ----------------------------
# Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    warmup_steps=500,
    save_steps=args.save_steps,
    save_total_limit=2,
    eval_strategy="steps",         # Évaluation régulière
    eval_steps=args.save_steps,          # Même fréquence que la sauvegarde
    logging_dir=os.path.join(args.output_dir, "logs"),
    logging_steps=50,
    load_best_model_at_end=True,
    save_strategy="steps",
    report_to=["wandb"],
)

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ----------------------------
# Train
# ----------------------------
print("🚀 Début de l'entraînement...")
trainer.train()

# ----------------------------
# Save model
# ----------------------------
print(f"💾 Sauvegarde du modèle dans {args.save_dir}...")
trainer.save_model(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
print(f"✅ Modèle et tokenizer sauvegardés.")
