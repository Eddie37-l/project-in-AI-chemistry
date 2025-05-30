import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np
import os

# === 1. Locate and load the CSV file ===
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "..", "..", "data_smiles", "ligand_candidates.csv")
file_path = os.path.normpath(file_path)


df = pd.read_csv(file_path)

# === 2. Function to calculate molecular weight from SMILES ===
def calc_mol_weight(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Descriptors.MolWt(mol)
        else:
            return None
    except:
        return None

df['Molecular_Weight'] = df['Ligand_SMILES'].apply(calc_mol_weight)

# === 3. Clean the data_output_pretraining (remove rows with SMILES parsing errors) ===
df_clean = df.dropna(subset=['Molecular_Weight'])

# === 4. Compute KDE (Gaussian Kernel Density Estimate) ===
weights = df_clean['Molecular_Weight'].values
kde = gaussian_kde(weights)
x_range = np.linspace(weights.min(), weights.max(), 1000)
density = kde(x_range)

# Define a density threshold (e.g., 50% of the max density)
threshold = 0.5 * max(density)

# Keep only values within the dense region
dense_region = x_range[density >= threshold]
lower_bound = dense_region.min()
upper_bound = dense_region.max()

print(f"Density-based range (50% max): {lower_bound:.2f} – {upper_bound:.2f} g/mol")

# === 5. Plot KDE distribution with selected range ===
plt.figure(figsize=(10, 6))
sns.histplot(weights, kde=True, bins=15, stat="density")
plt.fill_between(x_range, 0, density, where=(density >= threshold), color='orange', alpha=0.3, label='Useful zone (density ≥ 50%)')
plt.axvline(lower_bound, color='blue', linestyle='--', label=f'Lower bound = {lower_bound:.2f}')
plt.axvline(upper_bound, color='blue', linestyle='--', label=f'Upper bound = {upper_bound:.2f}')
plt.title("Estimated Gaussian Distribution of Molecular Weights")
plt.xlabel("Molecular Weight (g/mol)")
plt.ylabel("Estimated Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






