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

metaux_a_supprimer = {"Ag", "Cu", "Au", "Ni", "Pd", "Pt"}

# === Créer le dossier de sortie ===
os.makedirs(output_dir, exist_ok=True)

# === Supprimer le fichier de sortie s'il existe déjà ===
if os.path.exists(output_csv_file):
    os.remove(output_csv_file)
    print(f"🧹 Ancien fichier supprimé : {output_csv_file}")

# === Initialiser le nouveau fichier CSV avec en-tête ===
with open(output_csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])

# === Fonction : nettoyer et filtrer une molécule ===
def nettoyer_et_filtrer(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # Supprimer les métaux
    rw_mol = RWMol(mol)
    indices = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() in metaux_a_supprimer]
    for idx in sorted(indices, reverse=True):
        rw_mol.RemoveAtom(idx)
    mol = rw_mol.GetMol()

    # Vérifier la validité
    try:
        Chem.SanitizeMol(mol)
    except:
        return None

    # Recalculer descripteurs
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

# === Partie 1 : Dataset Hugging Face ===
print(f"\n📡 Traitement du dataset : {datasets_to_use[0][0]}")
ds = load_dataset(datasets_to_use[0][0], split="train", streaming=True)

total_hf = 0
with open(output_csv_file, "a", newline='') as f:
    writer = csv.writer(f)
    for row in ds:
        raw_smi = row.get(datasets_to_use[0][1], "").strip()
        cleaned_smi = nettoyer_et_filtrer(raw_smi)
        if cleaned_smi:
            writer.writerow([cleaned_smi])
            total_hf += 1

print(f"✅ {total_hf} SMILES écrits depuis Hugging Face")

# === Partie 2 : Fichier local Cross_Coupling_Smiles.csv ===
print(f"\n📂 Traitement du fichier local : {input_csv_file}")
total_local = 0

if os.path.exists(input_csv_file):
    with open(input_csv_file, "r", newline='') as f_in, open(output_csv_file, "a", newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        for row in reader:
            raw_smi = row.get("SMILES", "").strip()
            cleaned_smi = nettoyer_et_filtrer(raw_smi)
            if cleaned_smi:
                writer.writerow([cleaned_smi])
                total_local += 1
    print(f"✅ {total_local} SMILES ajoutés depuis Cross_Coupling_Smiles.csv")
else:
    print(f"⚠️ Fichier non trouvé : {input_csv_file}")

# === Résumé final ===
print(f"\n🎉 Total combiné : {total_hf + total_local} molécules filtrées et nettoyées")


