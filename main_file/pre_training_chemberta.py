import argparse
import os
import wandb
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_from_disk, DatasetDict

# ----------------------------
# Argument parsing
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="datasets 'train' and 'val'")
parser.add_argument("--output_dir", type=str, default="output/chemberta-mlm3", help="results' output")
parser.add_argument("--save_dir", type=str, default="models/chemberta-mlm-custom3", help="where to save the model")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=5, help="epochs number")
parser.add_argument("--save_steps", type=int, default=500, help="checkpoint/save frequency in steps")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of steps for gradient accumulation")
args = parser.parse_args()

# ----------------------------
# Init WandB
# ----------------------------
wandb.init(project="chemberta-pretraining3", name="ChemBERTa MLM Custom3")

# ----------------------------
# Load tokenizer and model
# ----------------------------
print("🔹 loading tokenizer/model...")
tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
print("✅ Tokenizer loaded.")

model = RobertaForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
print("✅ Model charged.")

# ----------------------------
# Load dataset
# ----------------------------
print(f"🔹 charging of the dataset from {args.dataset_dir}...")
train_dataset = load_from_disk(os.path.join(args.dataset_dir, "train"))
val_dataset = load_from_disk(os.path.join(args.dataset_dir, "val"))
dataset = DatasetDict({"train": train_dataset,"validation": val_dataset})
print(f"✅ Dataset loaded. Train : {len(dataset['train'])}, Validation : {len(dataset['validation'])}")

# ----------------------------
# Data collator
# ----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# ----------------------------
# Calculate total training steps for warmup
# ----------------------------
train_data_len = len(dataset["train"])
effective_batch_size = args.batch_size * args.gradient_accumulation_steps
total_training_steps = (train_data_len // effective_batch_size) * args.epochs
warmup_steps = max(100, int(0.05 * total_training_steps))

print(f"Total training steps: {total_training_steps}, Warmup steps: {warmup_steps}")

# ----------------------------
# Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    warmup_steps=warmup_steps,
    save_steps=args.save_steps,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=args.save_steps,
    logging_dir=os.path.join(args.output_dir, "logs"),
    logging_steps=50,
    load_best_model_at_end=True,
    save_strategy="steps",
    report_to=["wandb"],
    run_name= "ChemBERTa MLM Custom3",
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
print("🚀 Training as started...")
trainer.train()

# ----------------------------
# Save model
# ----------------------------
print(f"💾 Saving to {args.save_dir}...")
trainer.save_model(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
print(f"✅ Training finished.")
