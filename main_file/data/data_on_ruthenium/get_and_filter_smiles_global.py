from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import csv

# === CONFIGURATION ===
input_file = os.path.join(os.path.dirname(__file__), "smiles.txt")
output_dir = "data_smiles"
output_csv_file = os.path.join(output_dir, "filtered_smiles_global.csv")

# === Préparer le dossier de sortie ===
os.makedirs(output_dir, exist_ok=True)

# === Lire les SMILES depuis le fichier texte ===
with open(input_file, "r") as f:
    smiles_list = [line.strip() for line in f if line.strip()]

# === Initialiser le fichier CSV avec en-tête ===
with open(output_csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])

# === Filtrage et écriture ===
filtered_count = 0
batch = []

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue

    mw = Descriptors.MolWt(mol)
    if 100 <= mw <= 800:
        batch.append([smi])
        filtered_count += 1

    if len(batch) >= 100:
        with open(output_csv_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(batch)
        batch = []

# Derniers restes
if batch:
    with open(output_csv_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(batch)

print(f"\n✅ {filtered_count} SMILES filtrés enregistrés dans {output_csv_file}")










