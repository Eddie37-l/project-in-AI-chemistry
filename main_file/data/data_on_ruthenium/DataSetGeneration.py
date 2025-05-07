import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Charger le fichier contenant les SMILES
df = pd.read_csv("Ligand_candidates.csv")
df.columns = df.columns.str.strip()  # Corrige les noms de colonnes s’ils ont des espaces

# Nettoyer les SMILES
df['Mol'] = df['Ligand_SMILES'].apply(lambda x: Chem.MolFromSmiles(x) if pd.notnull(x) else None)
df = df[df['Mol'].notnull()]

# Calculer les propriétés chimiques
df['MolWt'] = df['Mol'].apply(Descriptors.MolWt)
df['NumC'] = df['Mol'].apply(lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'))
df['NumP'] = df['Mol'].apply(lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'P'))
df['NumN'] = df['Mol'].apply(lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'))

# Filtrer par masse molaire
filtered = df[(df['MolWt'] >= 180) & (df['MolWt'] <= 575)]

# Moyenne du nombre d'atomes
avg_C = round(filtered['NumC'].mean())
avg_P = round(filtered['NumP'].mean())
avg_N = round(filtered['NumN'].mean())

print(f"Moyennes : C={avg_C}, P={avg_P}, N={avg_N}")
