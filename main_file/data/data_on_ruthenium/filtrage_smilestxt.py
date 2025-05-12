from rdkit import Chem
from rdkit.Chem import Descriptors

# Charger les SMILES depuis le fichier
with open("smiles.txt", "r") as f:
    smiles_list = [line.strip() for line in f if line.strip()]

# Stocker les SMILES qui respectent les critères
filtered_smiles = []

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"❌ SMILES invalide ignoré : {smi}")
        continue

    # Calculer la masse molaire
    mw = Descriptors.MolWt(mol)

    # Compter les atomes
    atom_counts = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

    # Appliquer les critères
    if (
        230 <= mw <= 593 and
        20 <= atom_counts.get("C", 0) <= 25 and
        atom_counts.get("P", 0) == 1 and
        atom_counts.get("N", 0) == 1
    ):
        filtered_smiles.append(smi)

# Enregistrer les résultats
with open("filtered_smiles.txt", "w") as f:
    for smi in filtered_smiles:
        f.write(smi + "\n")

print(f"✅ {len(filtered_smiles)} SMILES retenus et sauvegardés dans 'filtered_smiles.txt'")



