# RuBERT: Yield Prediction with ChemBERTa for Ruthenium-Catalyzed Hydrogenation of carbonyl compounds

This project explores the use of a ChemBERTa-based model, called **RuBERT**, to predict reaction yields in **ruthenium-catalyzed hydrogenation** of carbonyl compounds. It combines machine learning on chemical SMILES and experimental data from literature.

A MODIFIER !!!!
---

## 📁 Project Structure

- `main_file/`: Core scripts for model training, data filtering, and tokenization
- `data_smiles/`: Contains ligand SMILES and filtered datasets
- `README.md`: This file

---

## 🚀 How to Run


### ✅ Requirements

Make sure you have Python 3.10 and install dependencies:

```bash
conda create -n chemberta_env python=3.10
conda activate chemberta_env
pip install -r requirements.txt
```

## 📊 Key Features

- ✅ **ChemBERTa-based model (RuBERT)** for predicting yields in Ru-catalyzed hydrogenation reactions
- ✅ **Custom SMILES filtering** based on molecular weight of 30 phosphine ligands, atom content (P & N), and absence of metals
- ✅ **Visualization tools** for ligand distribution (KDE-based selection)
- ✅ **Modular scripts** for preprocessing, model training, and ligand analysis
- ✅ **Integration of multiple data sources**: Hugging Face datasets + MPCD + Cross-coupling datasets (Schwaller, 2023)

---

## 📦 Dataset Sources

- **Hugging Face**: [`antoinebcx/smiles-molecules-chembl`](https://huggingface.co/datasets/antoinebcx/smiles-molecules-chembl)
- **Local**: `data_smiles/Cross_Coupling_Smiles.csv`, `.xyz` ligand files from `structures_all/`, `structures_all2/`
- **Ligand filtering** based on molecular weight distributions derived from `Ligand_candidates.csv`

---

## ✍️ Contributors

- Thomas Cohen - Pre-training, Package Infrastructure
- William Pellassy - Package Infrastructure, Dataset retrievement/filtering
- Edward Von Doderer -

---

