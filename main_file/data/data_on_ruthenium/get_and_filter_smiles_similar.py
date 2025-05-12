from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import csv

# === CONFIGURATION ===
datasets_to_use = [
    ("antoinebcx/smiles-molecules-chembl", "smiles"),
]
output_dir = "data_smiles"
output_csv_file = os.path.join(output_dir, "filtered_smiles_similar.csv")  # ✅ Changement ici

# === Préparer le dossier de sortie ===
os.makedirs(output_dir, exist_ok=True)

# === Initialiser le fichier CSV avec en-tête ===
with open(output_csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])  # En-tête

total_filtered = 0

for dataset_name, smiles_column in datasets_to_use:
    print(f"\n📡 Streaming dataset : {dataset_name}")
    ds = load_dataset(dataset_name, split="train", streaming=True)

    filtered_smiles = []

    for row in ds:
        smi = row.get(smiles_column, "").strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)
        atom_counts = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

        if (
            230 <= mw <= 593 and
            20 <= atom_counts.get("C", 0) <= 25 and
            atom_counts.get("P", 0) == 1 and
            atom_counts.get("N", 0) == 1
        ):
            filtered_smiles.append([smi])
            total_filtered += 1

        # Écriture progressive pour éviter les pertes
        if len(filtered_smiles) >= 100:
            with open(output_csv_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(filtered_smiles)
            filtered_smiles = []

    # Écrire les derniers restants
    if filtered_smiles:
        with open(output_csv_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(filtered_smiles)

print(f"\n🎉 Total final : {total_filtered} SMILES écrits dans '{output_csv_file}'")

