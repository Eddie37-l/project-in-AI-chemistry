import pandas as pd

# charge the database with the SMILES
df = pd.read_csv("../data/dara_for_pretraining/filtered_smiles_similar.csv")

# keep only the ligand column
ligands = df['smiles'].dropna()

# Save the file
with open("../data/dara_for_pretraining/filtred_ligands_raw.txt", "w") as f:
    for ligand in ligands:
        f.write(ligand.strip() + "\n")

print(f"{len(ligands)} saved ligands.")
