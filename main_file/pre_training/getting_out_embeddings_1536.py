import pandas as pd
import torch
from transformers import RobertaTokenizerFast, RobertaModel
from tqdm import tqdm
import numpy as np

# --------------------------
# ⚙️ Configuration
# --------------------------
MODEL = "seyonec/ChemBERTa-zinc-base-v1"  # <- ton modèle pré-entraîné
CSV_INPUT_PATH = "../data/dara_for_pretraining/data_hydrogenation_tokenized_2.csv"  # <- ton fichier d'entrée
CSV_OUTPUT_PATH = "data_output_pretraining/embeddings_output_modelbrut.csv"  # <- fichier de sortie
USE_CUDA = torch.cuda.is_available()

# --------------------------
# 🔹 Load model/tokenizer
# --------------------------
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL)
model = RobertaModel.from_pretrained(MODEL)
model.eval()
if USE_CUDA:
    model.cuda()

# --------------------------
# 🔹 Load the CSV file
# --------------------------
df = pd.read_csv(CSV_INPUT_PATH)

# --------------------------
# 🔹 Embedding function
# --------------------------
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    if USE_CUDA:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding  # shape: (768,)

# --------------------------
# 🔹 Extract embeddings for ligand and substrate separately
# --------------------------
embeddings = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔄 Extraction embeddings 1536"):
    ligand_emb = get_embedding(str(row["Ligand_SMILES"]))
    substrate_emb = get_embedding(str(row["Substrate_SMILES"]))
    combined_emb = np.concatenate([ligand_emb, substrate_emb])  # shape: (1536,)
    embeddings.append(combined_emb)

# --------------------------
# 🔹 Convert to DataFrame
# --------------------------
emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(1536)])
emb_df["Entry"] = df["Entry"]
emb_df["Yield"] = df["Yield"]

# --------------------------
# 🔹 Save to CSV
# --------------------------
emb_df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"✅ Embeddings 1536 dimensions exportés vers : {CSV_OUTPUT_PATH}")
