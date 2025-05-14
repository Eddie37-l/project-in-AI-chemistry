import pandas as pd

# charge the database with the SMILES
df = pd.read_csv("data/data_on_ruthenium/data_hydrogenation.csv")

# keep only the ligand column
ligands = df['Ligand_SMILES'].dropna()

# Save the file
with open("data/ligands_raw.txt", "w") as f:
    for ligand in ligands:
        f.write(ligand.strip() + "\n")

print(f"{len(ligands)} saved ligands.")
