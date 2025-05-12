from datasets import load_dataset

ds = load_dataset("di-zhang-fdu/ChemData700K-SMILES-only")

# Preview the data
print(ds['train'][0])
