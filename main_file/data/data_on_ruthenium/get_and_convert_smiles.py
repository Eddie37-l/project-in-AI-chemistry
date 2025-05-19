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
metaux_a_supprimer = {"Ag", "Cu", "Au", "Ni", "Pd", "Pt"}

# === 4 dossiers contenant les fichiers .xyz ===
xyz_root_paths = [
    os.path.join("main_file", "data", "data_on_ruthenium", "structures_all", "structures_1", "All2Lig"),
    os.path.join("main_file", "data", "data_on_ruthenium", "structures_all", "structures_1", "All4Lig"),
    os.path.join("main_file", "data", "data_on_ruthenium", "structures_all2", "structures_2", "All2Lig"),
    os.path.join("main_file", "data", "data_on_ruthenium", "structures_all2", "structures_2", "All4Lig"),
]

# === Préparation dossier de sortie ===
os.makedirs(output_dir, exist_ok=True)
if os.path.exists(output_csv_file):
    os.remove(output_csv_file)
    print(f"🧹 Ancien fichier supprimé : {output_csv_file}")

# === Réduction des logs Open Babel ===
openbabel.obErrorLog.SetOutputLevel(0)

# === Fonction de nettoyage / filtrage ===
def nettoyer_et_filtrer(mol):
    if mol is None:
        return None
    try:
        rw_mol = RWMol(mol)
        indices = [a.GetIdx() for a in rw_mol.GetAtoms() if a.GetSymbol() in metaux_a_supprimer]
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

    if 50 <= mw <= 500 and atom_counts.get("P", 0) >= 1 and atom_counts.get("N", 0) >= 1:
        return Chem.MolToSmiles(mol)
    return None

# === COLLECTION DES SMILES UNIQUES ===
smiles_uniques = set()
total_xyz = 0
total_hf = 0

# === 📡 Partie Hugging Face ===
print(f"\n📡 Streaming depuis Hugging Face : {huggingface_dataset[0]}")
ds = load_dataset(huggingface_dataset[0], split="train", streaming=True)

for row in ds:
    smi = row.get(huggingface_dataset[1], "").strip()
    mol = Chem.MolFromSmiles(smi)
    cleaned = nettoyer_et_filtrer(mol)
    if cleaned and cleaned not in smiles_uniques:
        smiles_uniques.add(cleaned)
        total_hf += 1

print(f"✅ {total_hf} SMILES ajoutés depuis Hugging Face")

# === 📂 Partie .xyz ===
print(f"\n📂 Analyse des fichiers .xyz :")
for path in xyz_root_paths:
    print(f"🔍 Dossier : {path}")
    if not os.path.exists(path):
        print(f"⛔ Dossier introuvable : {path}")
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
                    raise ValueError("SMILES vide (conversion échouée)")
                smi = smi_raw.split()[0]
                rdkit_mol = Chem.MolFromSmiles(smi)
                cleaned = nettoyer_et_filtrer(rdkit_mol)

                if cleaned and cleaned not in smiles_uniques:
                    smiles_uniques.add(cleaned)
                    total_xyz += 1
            except Exception as e:
                print(f"⚠️ Échec conversion : {file_path} → {e}")

# === ÉCRITURE FINALE ===
with open(output_csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])
    for smi in sorted(smiles_uniques):
        writer.writerow([smi])

print(f"\n✅ Terminé : {len(smiles_uniques)} SMILES uniques écrits dans '{output_csv_file}'")
print(f"🔹 Dont {total_hf} depuis Hugging Face, {total_xyz} depuis les fichiers .xyz")



