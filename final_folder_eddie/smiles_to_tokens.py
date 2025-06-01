import pandas as pd
from transformers import AutoTokenizer

model_name = "chemberta-mlm-custom-3" #choose your model here for your wanted tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_two_smiles(
    data: str,
    low_lim: int,
    high_lim: int,
    pad: bool,
    trunc: bool,
    max_len: int,
    yield_col: str
) -> str:
    """
    This file will convert a csv file with smiles to a csv file with tokenized smiles.It also
    returns the attentiuon mask for both. You can choose all your tokenization parameters below.

    """


    df = pd.read_csv(data)


    if yield_col not in df.columns:
        raise ValueError(f"Could not find '{yield_col}' column in '{data}'.")


    smi_columns = list(df.columns[low_lim:high_lim])
    if len(smi_columns) != 2:
        raise ValueError(f"Expected exactly two columns in the range {low_lim}:{high_lim}, got {len(smi_columns)}.")


    df_out = df[[yield_col]].copy()


    for col in smi_columns:
        all_input_ids = []
        all_attention_masks = []

        for text in df[col].fillna("").astype(str):
            enc = tokenizer(
                text,
                padding="max_length" if pad else False,
                truncation=trunc,
                max_length=max_len,
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].squeeze(0).tolist()       # length == max_len
            attn_mask = enc["attention_mask"].squeeze(0).tolist()  # length == max_len


            all_input_ids.append(" ".join(str(t) for t in input_ids))
            all_attention_masks.append(" ".join(str(m) for m in attn_mask))


        df_out[f"{col}_input_ids"] = all_input_ids
        df_out[f"{col}_attention_mask"] = all_attention_masks

    # 6) Save out only: [Yield, smi1_input_ids, smi1_attention_mask, smi2_input_ids, smi2_attention_mask]
    out_filename = data.rsplit(".", 1)[0] + "_tokenized.csv"
    df_out.to_csv(out_filename, index=False)
    return out_filename


if __name__ == "__main__":

    output_file = tokenize_two_smiles(
        data="data_hydrogenation_smiles.csv",
        low_lim=1,           # you use low_lim and high_lim to choose which colums you want to tokenize
        high_lim=3,
        pad=True,
        trunc=True,
        max_len=110,        #this is where you choose the max lenth of the tokens 110 was chose as it takes in all our smiles
        yield_col="Yield"    # name of the column you want to keep (capital Y)
    )
    print(f"Saved tokenized CSV to: {output_file}")



