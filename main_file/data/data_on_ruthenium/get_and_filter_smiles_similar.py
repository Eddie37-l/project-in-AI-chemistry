from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem import Descriptors, RWMol
import os
import csv

# === CONFIGURATION ===
datasets_to_use = [("antoinebcx/smiles-molecules-chembl", "smiles")]
output_dir = "data_smiles"
output_csv_file = os.path.join(output_dir, "filtered_smiles_similar.csv")
input_csv_file = os.path.join(output_dir, "Cross_Coupling_Smiles.csv")

metals_to_remove = {"Ag", "Cu", "Au", "Ni", "Pd", "Pt"}

# === Create output directory ===
os.makedirs(output_dir, exist_ok=True)

# === Delete output file if it already exists ===
if os.path.exists(output_csv_file):
    os.remove(output_csv_file)
    print(f"🧹 Old file deleted: {output_csv_file}")

# === Initialize new CSV file with header ===
with open(output_csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])

# === Function: clean and filter a molecule ===
def clean_and_filter(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # Remove metals
    rw_mol = RWMol(mol)
    indices = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() in metals_to_remove]
    for idx in sorted(indices, reverse=True):
        rw_mol.RemoveAtom(idx)
    mol = rw_mol.GetMol()

    # Check validity
    try:
        Chem.SanitizeMol(mol)
    except:
        return None

    # Recompute descriptors
    mw = Descriptors.MolWt(mol)
    atom_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1

    if (
        50 <= mw <= 1000 and
        atom_counts.get("P", 0) >= 1 and
        atom_counts.get("N", 0) >= 1
    ):
        return Chem.MolToSmiles(mol)
    return None

# === Part 1: Hugging Face dataset ===
print(f"\n📡 Processing dataset: {datasets_to_use[0][0]}")
ds = load_dataset(datasets_to_use[0][0], split="train", streaming=True)

total_hf = 0
with open(output_csv_file, "a", newline='') as f:
    writer = csv.writer(f)
    for row in ds:
        raw_smi = row.get(datasets_to_use[0][1], "").strip()
        cleaned_smi = clean_and_filter(raw_smi)
        if cleaned_smi:
            writer.writerow([cleaned_smi])
            total_hf += 1

print(f"✅ {total_hf} SMILES written from Hugging Face")

# === Part 2: Local file Cross_Coupling_Smiles.csv ===
print(f"\n📂 Processing local file: {input_csv_file}")
total_local = 0

if os.path.exists(input_csv_file):
    with open(input_csv_file, "r", newline='') as f_in, open(output_csv_file, "a", newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        for row in reader:
            raw_smi = row.get("SMILES", "").strip()
            cleaned_smi = clean_and_filter(raw_smi)
            if cleaned_smi:
                writer.writerow([cleaned_smi])
                total_local += 1
    print(f"✅ {total_local} SMILES added from Cross_Coupling_Smiles.csv")
else:
    print(f"⚠️ File not found: {input_csv_file}")

# === Final summary ===
print(f"\n🎉 Combined total: {total_hf + total_local} molecules filtered and cleaned")



