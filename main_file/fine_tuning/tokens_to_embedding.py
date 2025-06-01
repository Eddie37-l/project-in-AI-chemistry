import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModel
"""
    converts tokens to 2 embeddings, was used to create the data for 
    the fine tuning 

    """
# Configuration
MODEL_NAME    = "chemberta-mlm-custom-3" #choose your model here
OUTPUT_DIR    = "embeddings_output"
BATCH_SIZE    = 32

# The exact column names in your tokenized CSV:
YIELD_COL                      = "Yield"
LIGAND_IDS_COL                 = "Ligand_SMILES_input_ids"
LIGAND_MASK_COL                = "Ligand_SMILES_attention_mask"
SUBSTRATE_IDS_COL              = "Substrate_SMILES_input_ids"
SUBSTRATE_MASK_COL             = "Substrate_SMILES_attention_mask"

# Initialize model (we don’t need the tokenizer because we already have IDs/masks)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
# If you have a GPU and want to speed up:
# model = model.to("cuda")


def parse_space_separated_ints(str_of_ints: str) -> list[int]:
    """
    Convert a single string like "101 102 103 104 ..." into a List[int].
    If the string is empty or whitespace, returns an empty list.
    """
    if not isinstance(str_of_ints, str) or str_of_ints.strip() == "":
        return []
    return [int(x) for x in str_of_ints.split()]


def tokenized_to_embeddings(
    all_input_ids: list[list[int]],
    all_attention_masks: list[list[int]],
) -> torch.Tensor:
    """
    Given two Python lists-of-lists (each inner list is length = seq_len),
    batch them into chunks of size BATCH_SIZE, run through ChemBERTa, and return
    a torch.Tensor of shape (N, hidden_size) containing each row’s [CLS] embedding.

    - all_input_ids:   [ [id_0, id_1, …, id_{L-1}],  # row 0
                         [id_0, id_1, …, id_{L-1}],  # row 1
                         …                       ]   # row N-1
      where L = fixed max_len (e.g. 110)
    - all_attention_masks: same shape, but values in {0,1}.
    """
    embeddings: list[torch.Tensor] = []
    N = len(all_input_ids)
    assert N == len(all_attention_masks), "Number of input_ids lists must match number of attention_mask lists."

    # Convert entire dataset to torch tensors on CPU (or GPU if moved above)
    input_ids_tensor      = torch.tensor(all_input_ids,      dtype=torch.long)
    attention_mask_tensor = torch.tensor(all_attention_masks, dtype=torch.long)

    for start in tqdm(range(0, N, BATCH_SIZE), desc="Embedding batches"):
        end = min(start + BATCH_SIZE, N)
        batch_input_ids = input_ids_tensor[start:end]         # shape (batch_size, seq_len)
        batch_attention = attention_mask_tensor[start:end]    # shape (batch_size, seq_len)

        # If you moved model to GPU, also move these:
        # batch_input_ids = batch_input_ids.to("cuda")
        # batch_attention = batch_attention.to("cuda")

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention
            )
        # outputs.last_hidden_state: (batch_size, seq_len, hidden_size)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        embeddings.append(cls_emb.cpu())

    return torch.cat(embeddings, dim=0)  # final shape: (N, hidden_size)


def convert_tokenized_csv_to_embedding_file(
    input_csv: str,
    output_csv: str | None = None
) -> str:
    """
    Reads a *tokenized* CSV with EXACT columns:
        - Yield
        - Ligand_SMILES_input_ids
        - Ligand_SMILES_attention_mask
        - Substrate_SMILES_input_ids
        - Substrate_SMILES_attention_mask

    Parses those space‐separated integer strings back into Python lists,
    computes [CLS] embeddings for each row of Ligand and Substrate,
    and writes out a new CSV containing:
      • Yield
      • Ligand_SMILES_emb_0 ... Ligand_SMILES_emb_{hidden_size-1}
      • Substrate_SMILES_emb_0 ... Substrate_SMILES_emb_{hidden_size-1}

    Returns the path to the newly created CSV.
    """
    df = pd.read_csv(input_csv)

    # Ensure all required columns are present
    required = [
        YIELD_COL,
        LIGAND_IDS_COL,
        LIGAND_MASK_COL,
        SUBSTRATE_IDS_COL,
        SUBSTRATE_MASK_COL,
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in '{input_csv}'.")

    # 1) Build the “side” of the output DataFrame (just Yield)
    out_df = pd.DataFrame({YIELD_COL: df[YIELD_COL]})

    # 2) Parse Ligand IDs & mask → embeddings
    ligand_ids_lists = df[LIGAND_IDS_COL].fillna("").map(parse_space_separated_ints).tolist()
    ligand_mask_lists = df[LIGAND_MASK_COL].fillna("").map(parse_space_separated_ints).tolist()
    ligand_emb_tensor = tokenized_to_embeddings(ligand_ids_lists, ligand_mask_lists)
    ligand_emb_array = ligand_emb_tensor.numpy()  # shape: (N, hidden_size)
    n_hidden = ligand_emb_array.shape[1]

    # Create DataFrame for ligand embeddings
    ligand_feature_names = [f"Ligand_SMILES_emb_{i}" for i in range(n_hidden)]
    ligand_emb_df = pd.DataFrame(ligand_emb_array, columns=ligand_feature_names)
    out_df = pd.concat([out_df.reset_index(drop=True), ligand_emb_df], axis=1)

    # 3) Parse Substrate IDs & mask → embeddings
    substrate_ids_lists = df[SUBSTRATE_IDS_COL].fillna("").map(parse_space_separated_ints).tolist()
    substrate_mask_lists = df[SUBSTRATE_MASK_COL].fillna("").map(parse_space_separated_ints).tolist()
    substrate_emb_tensor = tokenized_to_embeddings(substrate_ids_lists, substrate_mask_lists)
    substrate_emb_array = substrate_emb_tensor.numpy()  # shape: (N, hidden_size)

    # Create DataFrame for substrate embeddings
    substrate_feature_names = [f"Substrate_SMILES_emb_{i}" for i in range(n_hidden)]
    substrate_emb_df = pd.DataFrame(substrate_emb_array, columns=substrate_feature_names)
    out_df = pd.concat([out_df.reset_index(drop=True), substrate_emb_df], axis=1)

    # 4) Write to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_csv is None:
        base = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(OUTPUT_DIR, f"{base}_embeddings.csv")

    out_df.to_csv(output_csv, index=False)
    print(f"Saved embeddings to: {output_csv}")
    return output_csv


if __name__ == "__main__":

    convert_tokenized_csv_to_embedding_file(
        input_csv="data_pretrained_TOKENS.csv"
    )
