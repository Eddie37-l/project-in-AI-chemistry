import argparse
import math
import os
import torch

from datasets import load_from_disk
from transformers import (
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# ----------- arguments CLI --------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", required=True, help="File that contain the under folder test")
parser.add_argument("--model_dir", required=True, help="File of the pre-trained model")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
parser.add_argument("--output_dir", default="output/test_eval", help="Where to write local logs")
parser.add_argument("--wandb_project", default=None, help="If you want to log to W&B")
args = parser.parse_args()

# ----------- Loading of the model + tokenizer ----------------------
print("🔹 Loading model & tokenizer…")
tokenizer = RobertaTokenizerFast.from_pretrained(args.model_dir)
model = RobertaForMaskedLM.from_pretrained(args.model_dir)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model.to(device)
print("✅ Model on", device)

# ----------- loading dataset test ----------------------------
test_path = os.path.join(args.dataset_dir, "test")
print(f"🔹 Loading test set from {test_path} …")
ds_test = load_from_disk(test_path)
print(f"✅ Test set size : {len(ds_test)}")

# ----------- data_output_pretraining collator with dynamic masking --------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# ----------- TrainingArguments (for evaluation only) ----------------
eval_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_eval_batch_size=args.batch_size,
    do_train=False,
    do_eval=True,
    report_to="wandb",
)

# ----------- Trainer and evaluation ------------------------------
trainer = Trainer(
    model=model,
    args=eval_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("🚀 Evaluating on test set …")
metrics = trainer.evaluate(eval_dataset=ds_test)

eval_loss = metrics.get("eval_loss")
if eval_loss is None:
    print("debugging: eval_loss missing")
else:
    perplexity = math.exp(eval_loss) if eval_loss < 100 else float("inf")
    metrics["perplexity"] = perplexity
    print(f"📉  Eval loss      : {eval_loss:.4f}")
    print(f"🧠  Perplexity     : {perplexity:.2f}")

print("✅ Done.")
