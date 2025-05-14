import pandas as pd
from schwaller_tokenizer import smi_tokenizer

# Function security
def safe_tokenizer(smi):
    if pd.isna(smi):
        return ''
    try:
        return smi_tokenizer(smi)
    except AssertionError:
        print(f"Erreur de tokenization : {smi}")
        return ''

# pathway of the CSV.file
file_path = "data/data_on_ruthenium/data_hydrogenation.csv"

# read the file
df = pd.read_csv(file_path, sep=",")

# apply the tokenizer to the column
df['tokenized_ligand'] = df['Ligand_SMILES'].apply(safe_tokenizer)
df['tokenized_substrate'] = df['Substrate_SMILES'].apply(safe_tokenizer)

# overview
print(df[['Ligand_SMILES', 'tokenized_ligand', 'Substrate_SMILES', 'tokenized_substrate', 'Yield']].head())

# Sauvegarder dans un nouveau fichier
df.to_csv("data/data_on_ruthenium/data_hydrogenation_tokenized.csv", index=False)
