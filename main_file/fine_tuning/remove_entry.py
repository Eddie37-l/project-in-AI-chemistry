import os
import pandas as pd
"""
    Simple function to remove entries from the csv files

    """
# ─── EDIT THIS: put your CSV filename (in the same folder as this script) ───
INPUT_FILE = "tokens_77M_with_entry.csv"  # ← replace with your actual CSV filename

# ─── Derive an output filename by appending "_noEntry" before the .csv extension ───
base, ext = os.path.splitext(INPUT_FILE)
OUTPUT_FILE = f"{base}_noEntry{ext}"  # e.g. "data_with_entry_noEntry.csv"

# 1) Read the CSV from the current directory
df = pd.read_csv(INPUT_FILE)

# 2) If there is a column called "Entry", drop it. Otherwise do nothing.
if "Entry" in df.columns:
    df = df.drop(columns=["Entry"])
    print(f"Dropped 'Entry' from {INPUT_FILE}")
else:
    print(f"No 'Entry' column found in {INPUT_FILE}; nothing changed.")

# 3) Write the (possibly‐modified) DataFrame back to a new CSV in the same folder
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved result as {OUTPUT_FILE}")


