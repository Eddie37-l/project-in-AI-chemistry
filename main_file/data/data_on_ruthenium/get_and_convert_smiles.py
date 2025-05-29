import os
import csv
from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem import Descriptors, RWMol
from openbabel import pybel
import openbabel

# === CONFIGURATION ===
output_dir = "data_smiles"
output_csv_file = os.path.join(output_dir, "filtered_smiles_similar.csv")
huggingface_dataset = ("antoinebcx/smiles-molecules-chembl", "smiles")
metals_to_remove = {"Ag", "Cu", "Au", "Ni", "Pd", "Pt"}

# === 4 folders containing .xyz files ===
script_dir = os.path.dirname(os.path.abspath(__file__))

xyz_root_paths = [
    os.path.join(script_dir, "data_smiles", "structures_all", "structures_1", "All2Lig"),
    os.path.join(script_dir, "data_smiles", "structures_all", "structures_1", "All4Lig"),
    os.path.join(script_dir, "data_smiles", "structures_all2", "structures_2", "All2Lig"),
    os.path.join(script_dir, "data_smiles", "structures_all2", "structures_2", "All4Lig"),
]

# === Prepare output folder ===
os.makedirs(output_dir, exist_ok=True)
if os.path.exists(output_csv_file):
    os.remove(output_csv_file)
    print(f"🧹 Old file deleted: {output_csv_file}")

# === Reduce Open Babel logs ===
openbabel.obErrorLog.SetOutputLevel(0)

# === Cleaning / filtering function ===
def clean_and_filter(mol):
    if mol is None:
        return None
    try:
        rw_mol = RWMol(mol)
        indices = [a.GetIdx() for a in rw_mol.GetAtoms() if a.GetSymbol() in metals_to_remove]
        for idx in sorted(indices, reverse=True):
            rw_mol.RemoveAtom(idx)
        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol)
    except:
        return None

    mw = Descriptors.MolWt(mol)
    atom_counts = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

    if 50 <= mw <= 600 and atom_counts.get("P", 0) >= 1 and atom_counts.get("N", 0) >= 1:
        return Chem.MolToSmiles(mol)
    return None

# === COLLECT UNIQUE SMILES ===
unique_smiles = set()
total_xyz = 0
total_hf = 0

# === 📡 Hugging Face dataset ===
print(f"\n📡 Streaming from Hugging Face: {huggingface_dataset[0]}")
ds = load_dataset(huggingface_dataset[0], split="train", streaming=True)

for row in ds:
    smi = row.get(huggingface_dataset[1], "").strip()
    mol = Chem.MolFromSmiles(smi)
    cleaned = clean_and_filter(mol)
    if cleaned and cleaned not in unique_smiles:
        unique_smiles.add(cleaned)
        total_hf += 1

print(f"✅ {total_hf} SMILES added from Hugging Face")

# === 📂 Process .xyz files ===
print(f"\n📂 Processing .xyz files:")
for path in xyz_root_paths:
    print(f"🔍 Folder: {path}")
    if not os.path.exists(path):
        print(f"⛔ Folder not found: {path}")
        continue

    for dirpath, _, files in os.walk(path):
        for file in files:
            if not file.endswith(".xyz"):
                continue
            file_path = os.path.join(dirpath, file)

            try:
                mol = next(pybel.readfile("xyz", file_path))
                smi_raw = mol.write("smi").strip()
                if not smi_raw:
                    raise ValueError("Empty SMILES (conversion failed)")
                smi = smi_raw.split()[0]
                rdkit_mol = Chem.MolFromSmiles(smi)
                cleaned = clean_and_filter(rdkit_mol)

                if cleaned and cleaned not in unique_smiles:
                    unique_smiles.add(cleaned)
                    total_xyz += 1
            except Exception as e:
                print(f"⚠️ Conversion failed: {file_path} → {e}")

# === FINAL WRITE ===
with open(output_csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])
    for smi in sorted(unique_smiles):
        writer.writerow([smi])

print(f"\n✅ Done: {len(unique_smiles)} unique SMILES written to '{output_csv_file}'")
print(f"🔹 Including {total_hf} from Hugging Face, {total_xyz} from .xyz files")




