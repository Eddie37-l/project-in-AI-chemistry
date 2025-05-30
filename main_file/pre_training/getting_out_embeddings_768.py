import pandas as pd
import torch
from transformers import RobertaTokenizerFast, RobertaModel
from tqdm import tqdm

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "../models/chemberta-mlm-custom-3"
CSV_INPUT_PATH = "../data/dara_for_pretraining/data_hydrogenation_tokenized_2.csv"
CSV_OUTPUT_PATH = "data_output_pretraining/embeddings_output_768.csv"
USE_CUDA = torch.cuda.is_available()

# --------------------------
# Load model/tokenizer
# --------------------------
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)
model = RobertaModel.from_pretrained(MODEL_PATH)
model.eval()
if USE_CUDA:
    model.cuda()

# --------------------------
# charge the file
# --------------------------
df = pd.read_csv(CSV_INPUT_PATH)

#Concat smiles into two column if needed
def concat_smiles(row):
    return f"{row['Ligand_SMILES']} {row['Substrate_SMILES']}"
df["input"] = df.apply(concat_smiles, axis=1)

# --------------------------
# Fonction for getting out the embeddings
# --------------------------
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    if USE_CUDA:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Option 1 : CLS token (position 0)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding

# --------------------------
# Generation of embeddings
# --------------------------
embeddings = []
for text in tqdm(df["input"], desc="🔄 Extraction of embeddings"):
    emb = get_embedding(text)
    embeddings.append(emb)

# --------------------------
# Conversion into DataFrame
# --------------------------
emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(len(embeddings[0]))])

# Add column
emb_df["Entry"] = df["Entry"]
emb_df["Yield"] = df["Yield"]

# --------------------------
# Export final
# --------------------------
emb_df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"✅ Embeddings exported into : {CSV_OUTPUT_PATH}")
