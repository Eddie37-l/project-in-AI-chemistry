import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter

# === Charger le fichier CSV ===
file_path = "c:/Users/wpell/python-course/projects/project-in-AI-chemistry/project-in-AI-chemistry/main_file/data/data_on_ruthenium/Ligand_candidates.csv"

#file_path = "Ligand_candidates.csv"  Assure-toi que ce fichier est dans le même dossier que ce script
df = pd.read_csv(file_path)

# === Fonction pour calculer la masse molaire ===
def calc_mol_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Descriptors.MolWt(mol)
    return None

# === Fonction pour compter les atomes C, H, N, O, P, S ===
def count_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series([None]*6, index=['C_count', 'H_count', 'N_count', 'O_count', 'P_count', 'S_count'])
    atom_counts = Counter([atom.GetSymbol() for atom in mol.GetAtoms()])
    return pd.Series([
        atom_counts.get('C', 0),
        atom_counts.get('H', 0),
        atom_counts.get('N', 0),
        atom_counts.get('O', 0),
        atom_counts.get('P', 0),
        atom_counts.get('S', 0)
    ], index=['C_count', 'H_count', 'N_count', 'O_count', 'P_count', 'S_count'])

# === Appliquer les calculs ===
df['Molecular_Weight'] = df['Ligand_SMILES'].apply(calc_mol_weight)
atom_counts_df = df['Ligand_SMILES'].apply(count_atoms)
df = pd.concat([df, atom_counts_df], axis=1)

# === Nettoyer les données (enlever les lignes sans masse) ===
df_clean = df.dropna(subset=['Molecular_Weight'])

# === Statistiques sur la masse molaire ===
mean_weight = df_clean['Molecular_Weight'].mean()
std_weight = df_clean['Molecular_Weight'].std()
lower_bound = mean_weight - std_weight
upper_bound = mean_weight + std_weight

print(f"Moyenne : {mean_weight:.2f} g/mol")
print(f"Écart-type : {std_weight:.2f} g/mol")
print(f"Plage ±1σ : {lower_bound:.2f} – {upper_bound:.2f} g/mol")

# === Filtrage des ligands dans la plage ±1σ ===
df_filtered = df[
    (df['Molecular_Weight'] >= lower_bound) &
    (df['Molecular_Weight'] <= upper_bound)
]

# === Export CSV enrichi ===
df_filtered.to_csv("Ligands_filtrés_±1sigma.csv", index=False)
print("✅ Fichier exporté : Ligands_filtrés_±1sigma.csv")
