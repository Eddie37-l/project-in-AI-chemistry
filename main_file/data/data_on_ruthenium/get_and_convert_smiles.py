import os
import csv
from openbabel import pybel

output_dir = "data_smiles"
output_file = os.path.join(output_dir, "Cross_Coupling_Smiles.csv")
os.makedirs(output_dir, exist_ok=True)

smiles_data = []

for dirpath, _, filenames in os.walk("."):
    for file in filenames:
        if file.endswith(".xyz"):
            filepath = os.path.join(dirpath, file)
            try:
                mol = next(pybel.readfile("xyz", filepath))
                smiles = mol.write("smi").strip().split()[0]
                smiles_data.append([smiles, os.path.relpath(filepath)])
                print(f"✅ Converted: {filepath}")
            except Exception as e:
                print(f"⚠️ Erreur fichier {filepath} : {e}")

with open(output_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["SMILES", "Source File"])
    writer.writerows(smiles_data)

print(f"\n🎉 Terminé ! {len(smiles_data)} molécules enregistrées dans {output_file}")



