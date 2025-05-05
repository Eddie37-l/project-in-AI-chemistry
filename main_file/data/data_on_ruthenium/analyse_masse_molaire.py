import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Charger le fichier CSV ===
file_path = "Ligand_candidates.csv"  # Assure-toi que ce fichier est dans le même dossier
df = pd.read_csv(file_path)

# === 2. Fonction pour calculer la masse molaire depuis les SMILES ===
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

# === 3. Nettoyage des données (on enlève les lignes avec erreurs de parsing) ===
df_clean = df.dropna(subset=['Molecular_Weight'])

from scipy.stats import gaussian_kde
import numpy as np

# === 4. Calcul de la densité KDE (noyau gaussien) ===
weights = df_clean['Molecular_Weight'].values
kde = gaussian_kde(weights)
x_range = np.linspace(weights.min(), weights.max(), 1000)
density = kde(x_range)

# Seuil de densité (ex: 50% de la densité max)
threshold = 0.5 * max(density)

# Filtrer les valeurs de x où la densité est au-dessus du seuil
dense_region = x_range[density >= threshold]
lower_bound = dense_region.min()
upper_bound = dense_region.max()

print(f"Plage basée sur la densité (50% max): {lower_bound:.2f} – {upper_bound:.2f} g/mol")

# === 5. Plot de la distribution KDE + plage sélectionnée ===
plt.figure(figsize=(10, 6))
sns.histplot(weights, kde=True, bins=15, stat="density")
plt.fill_between(x_range, 0, density, where=(density >= threshold), color='orange', alpha=0.3, label='Zone utile (densité ≥ 50%)')
plt.axvline(lower_bound, color='blue', linestyle='--', label=f'Borne inférieure = {lower_bound:.2f}')
plt.axvline(upper_bound, color='blue', linestyle='--', label=f'Borne supérieure = {upper_bound:.2f}')
plt.title("Distribution gaussienne estimée des masses molaires")
plt.xlabel("Masse molaire (g/mol)")
plt.ylabel("Densité estimée")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 6. Filtrage basé sur cette plage et sauvegarde ===
df_filtered = df_clean[
    (df_clean['Molecular_Weight'] >= lower_bound) &
    (df_clean['Molecular_Weight'] <= upper_bound)
]
df_filtered.to_csv("Ligands_filtered_by_gaussian_density.csv", index=False)



