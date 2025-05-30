import pandas as pd
import torch
from transformers import RobertaTokenizerFast, RobertaModel
from tqdm import tqdm

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "../models/chemberta-mlm-custom-3"  # <- ton modèle pré-entraîné
CSV_INPUT_PATH = "../data/dara_for_pretraining/data_hydrogenation_tokenized_2.csv"  # <- ton fichier d'entrée
CSV_OUTPUT_PATH = "data_output_pretraining/embeddings_output_768.csv"  # <- fichier de sortie
USE_CUDA = torch.cuda.is_available()

# --------------------------
# 🔹 Load model/tokenizer
# --------------------------
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)
model = RobertaModel.from_pretrained(MODEL_PATH)
model.eval()
if USE_CUDA:
    model.cuda()

# --------------------------
# 🔹 Charger ton fichier
# --------------------------
df = pd.read_csv(CSV_INPUT_PATH)

# Concatène les deux colonnes de SMILES si nécessaire
def concat_smiles(row):
    return f"{row['Ligand_SMILES']} {row['Substrate_SMILES']}"

df["input"] = df.apply(concat_smiles, axis=1)

# --------------------------
# 🔹 Fonction d'extraction des embeddings
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
# 🔹 Générer tous les embeddings
# --------------------------
embeddings = []
for text in tqdm(df["input"], desc="🔄 Extraction des embeddings"):
    emb = get_embedding(text)
    embeddings.append(emb)

# --------------------------
# 🔹 Conversion en DataFrame
# --------------------------
emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(len(embeddings[0]))])

# Ajouter les colonnes utiles (index, yield, etc.)
emb_df["Entry"] = df["Entry"]
emb_df["Yield"] = df["Yield"]

# --------------------------
# 🔹 Export final
# --------------------------
emb_df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"✅ Embeddings exportés vers : {CSV_OUTPUT_PATH}")
