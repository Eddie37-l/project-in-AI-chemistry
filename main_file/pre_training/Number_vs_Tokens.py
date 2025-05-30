import os
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import numpy as np

# --- File path ---
file_path = os.path.join("data_smiles", "filtered_smiles_similar.csv")

# --- Tokenizer ---
def tokenize_smiles(smiles: str):
    """Simple character-level tokenizer for SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []  # Skip invalid molecules
    return list(smiles)

# --- Load dataset ---
def load_dataset(file_path: str):
    df = pd.read_csv(file_path)
    if "smiles" not in df.columns:
        raise ValueError("The column 'smiles' is not in the dataset.")
    return df["smiles"].dropna().tolist()

# --- Compute token lengths ---
def compute_token_lengths(smiles_list):
    token_lengths = []
    for smi in smiles_list:
        tokens = tokenize_smiles(smi)
        token_lengths.append(len(tokens))
    return token_lengths

# --- Plot with token threshold cutoff ---
def plot_with_token_threshold(token_lengths, threshold=100):
    # Calculate percentage of molecules below threshold
    count_below = sum(1 for t in token_lengths if t <= threshold)
    percent_below = (count_below / len(token_lengths)) * 100
    print(f"{percent_below:.2f}% of molecules have ≤ {threshold} tokens.")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold} tokens\n({percent_below:.2f}% ≤ this)')
    plt.xlabel("Number of Tokens")
    plt.ylabel("Number of Molecules")
    plt.title(f"Token Length Distribution with Threshold = {threshold} Tokens")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main execution ---
if __name__ == "__main__":
    smiles_list = load_dataset(file_path)
    token_lengths = compute_token_lengths(smiles_list)
    plot_with_token_threshold(token_lengths, threshold=90)  # Change 100 to your desired limit
