![RuBERT](https://github.com/Eddie37-l/project-in-AI-chemistry/blob/main/banner_RuBERT.png?raw=true)

# 🤖RuBERT: Yield Prediction with ChemBERTa for Ruthenium-Catalyzed Hydrogenation of carbonyl compounds


This project explores the use of a ChemBERTa-based model, called **RuBERT**, to predict reaction yields in **ruthenium-catalyzed hydrogenation** of carbonyl compounds. It combines machine learning on chemical SMILES and experimental data from literature.


---
## 📚 Table of Contents

- [🧠 Features](#features)
- [🚀 How to Run](#how-to-run)
- [🔧 Requirements](#requirements)
- [🗂️ Project Structure](#project-structure)
- [📦 Dataset Sources](#dataset-sources)
- [👥 Contributors](#contributors)
- [📄 Licence](#license)
  

---

###⚠️ MODIFIER FEATURES SI NECESSAIRE
## Features 

- 🤖 **RuBERT, a ChemBERTa-based model**:  
Predicts reaction yields in ruthenium-catalyzed hydrogenation of carbonyl compounds.

- 🧪 **Smart SMILES filtering**:  
Automatically selects ligands based on molecular weight, presence of phosphorus and nitrogen atoms, and absence of metals.

- 🛠️ **Modular and reusable scripts**:  
For data preprocessing, ligand analysis, and model training — easily adaptable for other reaction datasets.

---

## How to Run 
###⚠️ MODIFIER HOW TO RUN ET REQUIREMENTS

### Requirements 

Make sure you have Python 3.10 and install dependencies:

```bash
conda create -n chemberta_env python=3.10
conda activate chemberta_env
pip install -r requirements.txt
```
---

## Project Structure 

- `main_file/`: Core scripts for model training, data filtering, and tokenization
- `data_smiles/`: Contains ligand SMILES and filtered datasets

---

## Dataset Sources 

- 💻 **Hugging Face**: Extracted from [`antoinebcx/smiles-molecules-chembl`](https://huggingface.co/datasets/antoinebcx/smiles-molecules-chembl)
- 📁 **Cross-coupling Ligands**: `data_smiles/Cross_Coupling_Smiles.csv`, `.xyz` ligand files from `structures_all/`, `structures_all2/` extracted from [this study](https://doi.org/10.1039/D3DD00011C) by Schwaller *et al.*
- 📊 **Original Phosphine Ligands**: based on molecular weight distributions derived from `Ligand_candidates.csv` extracted from [this study](https://doi.org/10.1038/s41467-022-30718-x) by Nakajima *et al.*
- 🗒️ **MPCD**: `data_smiles/smiles.txt` extracted from MPCD

---

## Contributors 

###⚠️ MODIFIER PARTIE THOMAS ET EDDIE 
- 👨‍🔬 Thomas Cohen - Pre-training, Package Infrastructure
- 🧾 William Pellassy - Package Infrastructure, Dataset retrieval/filtering
- 👨‍💻 Edward Von Doderer - 

---
## License 

This project is licensed under the MIT License.
You are free to use, modify, and distribute this software with appropriate attribution.
See the [LICENSE](LICENSE) file for details.



