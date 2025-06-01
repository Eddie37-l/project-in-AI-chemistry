![RuBERT](https://github.com/Eddie37-l/project-in-AI-chemistry/blob/main/banner_RuBERT.png?raw=true)

# 🤖RuBERT: Yield Prediction with ChemBERTa for Ruthenium-Catalyzed Hydrogenation of carbonyl compounds


This project explores the use of a ChemBERTa-based model, called **RuBERT**, to predict reaction yields in **ruthenium-catalyzed hydrogenation** of carbonyl compounds. It combines machine learning on chemical SMILES and experimental data from literature.


---
## 📚 Table of Contents

- [🧠 Features](#features)
- [🔧 Requirements](#requirements)
- [🗂️ Project Structure](#project-structure)
- [📦 Dataset Sources](#dataset-sources)
- [👥 Contributors](#contributors)
- [📄 Licence](#license)
  

---

## Features 

- 🤖 **RuBERT, a ChemBERTa-based model**:  
Predicts reaction yields in ruthenium-catalyzed hydrogenation of carbonyl compounds.

- 🧪 **Smart SMILES filtering**:  
Automatically selects ligands based on molecular weight, presence of phosphorus and nitrogen atoms, and absence of metals.

- 🛠️ **Modular and reusable scripts**:  
For data preprocessing, ligand analysis, and model training; easily adaptable for other reaction datasets.

---

### Requirements 

Make sure you have Python 3.10 and install dependencies:

```bash
conda create -n chemberta_env python=3.10
conda activate chemberta_env
pip install -r requirements.txt
```
---

## Project Structure 

- `main_file/`: Core scripts for model training, fine-tuning, data and data filtering
  - `data/`: Contains all the SMILES data for pre-training and fine-tuning, as well as the script for data filtering
  - `pre_training/`: Contains all the scripts needed to pre-train a ChemBERTa model, as well as the resulting outputs
  - `fine_tuning/`: Contains scripts for fine-tuning the pre-trained model on the yield prediction task.
  - `models/`: Contains three pre-trained models. Although Model 1 performs best, it suffers from data leakage. Model 3 is slightly better than Model 2 and is considered the most reliable.

---

## Dataset Sources 

- 💻 **Hugging Face**: Extracted from [`antoinebcx/smiles-molecules-chembl`](https://huggingface.co/datasets/antoinebcx/smiles-molecules-chembl)  

- 📁 **Cross-coupling Ligands**: `data_smiles/Cross_Coupling_Smiles.csv`, `.xyz` ligand files from `structures_all/`, `structures_all2/` extracted from [this study](https://doi.org/10.1039/D3DD00011C) by Schwaller *et al.*

- 📊 **Original Phosphine Ligands**: based on molecular weight distributions derived from `Ligand_candidates.csv` extracted from [this study](https://doi.org/10.1038/s41467-022-30718-x) by Nakajima *et al.*

- 🗒️ **MPCD**: `data_smiles/smiles.txt` extracted from MPCD

---

## Contributors 

- 👨‍🔬 Thomas Cohen - Pre-training, Package Infrastructure
- 🧾 William Pellassy - Package Infrastructure, Dataset retrieval/filtering
- 👨‍💻 Edward Von Doderer - Fine-tunning

---
## License 

This project is licensed under the MIT License.
You are free to use, modify, and distribute this software with appropriate attribution.
See the [LICENSE](LICENSE) file for details.



