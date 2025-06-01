import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# ── 1) Improved SoftDecisionTree ──
class SoftDecisionTree(nn.Module):
    def __init__(self, input_dim: int, depth: int):
        super().__init__()
        self.depth = depth
        self.num_internal = 2 ** depth - 1
        self.num_leaves = 2 ** depth

        # Better initialization
        self.node_weights = nn.Parameter(
            torch.randn(self.num_internal, input_dim) * 0.1
        )
        self.node_biases = nn.Parameter(torch.zeros(self.num_internal))

        # Initialize leaf values in yield range [0, 100]
        self.leaf_values = nn.Parameter(
            torch.rand(self.num_leaves) * 100.0  # Random values between 0-100
        )

        # Build path mask
        mask = torch.zeros(self.num_leaves, self.num_internal)
        for leaf_idx in range(self.num_leaves):
            bits = format(leaf_idx, f"0{depth}b")
            for level, bit in enumerate(bits):
                if level == 0:
                    node_index = 0
                else:
                    prefix = int(bits[:level], 2)
                    node_index = (2 ** level - 1) + prefix
                mask[leaf_idx, node_index] = float(bit)

        self.register_buffer("path_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Compute routing probabilities
        logits = x @ self.node_weights.t() + self.node_biases
        logits = torch.clamp(logits, -10.0, 10.0)
        p = torch.sigmoid(logits)

        # Compute path probabilities
        p_expanded = p.unsqueeze(1)
        one_minus_p = (1 - p).unsqueeze(1)
        mask = self.path_mask.unsqueeze(0).expand(batch_size, -1, -1)

        p_select = mask * p_expanded + (1 - mask) * one_minus_p
        p_leaf = torch.prod(p_select, dim=2)

        # Normalize probabilities
        p_leaf = p_leaf + 1e-8
        p_leaf = p_leaf / p_leaf.sum(dim=1, keepdim=True)

        # Weighted prediction
        out = (p_leaf * self.leaf_values.unsqueeze(0)).sum(dim=1)
        return out


# ── 2) Fixed Embedding Model ──
class SimpleEmbeddingModel(nn.Module):
    def __init__(self, ligand_dim: int, substrate_dim: int,
                 hidden_dim: int = 383, tree_depth: int = 3):
        super().__init__()

        # Simple embedding processing
        self.ligand_net = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # FIX: Use substrate_dim for input, hidden_dim for output
        self.substrate_net = nn.Sequential(
            nn.Linear(substrate_dim, hidden_dim),  # Fixed: was substrate_dim
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)  # Fixed: was substrate_dim
        )

        # Simple fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Soft decision tree
        self.tree = SoftDecisionTree(
            input_dim=hidden_dim // 2,
            depth=tree_depth
        )

    def forward(self, ligand_emb: torch.Tensor, substrate_emb: torch.Tensor):
        # Process embeddings separately
        lig_out = self.ligand_net(ligand_emb)
        sub_out = self.substrate_net(substrate_emb)

        # Simple concatenation
        combined = torch.cat([lig_out, sub_out], dim=1)

        # Fusion
        fused = self.fusion(combined)

        # Tree prediction
        prediction = self.tree(fused)

        return prediction


# ── 3) Simple Dataset (no normalization) ──
class EmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Find embedding columns
        self.ligand_cols = sorted(
            [c for c in df.columns if c.startswith("Ligand_SMILES_emb_")],
            key=lambda name: int(name.split("_")[-1])
        )
        self.substrate_cols = sorted(
            [c for c in df.columns if c.startswith("Substrate_SMILES_emb_")],
            key=lambda name: int(name.split("_")[-1])
        )

        print(f"Found {len(self.ligand_cols)} ligand embedding dimensions")
        print(f"Found {len(self.substrate_cols)} substrate embedding dimensions")

        # Extract data - NO NORMALIZATION
        self.ligand_data = df[self.ligand_cols].values.astype(np.float32)
        self.substrate_data = df[self.substrate_cols].values.astype(np.float32)
        self.yields = df["Yield"].values.astype(np.float32)

    def __len__(self):
        return len(self.yields)

    def __getitem__(self, idx):
        return {
            "ligand_emb": torch.tensor(self.ligand_data[idx], dtype=torch.float32),
            "substrate_emb": torch.tensor(self.substrate_data[idx], dtype=torch.float32),
            "yield": torch.tensor(self.yields[idx], dtype=torch.float32)
        }


# ── 4) Training function ──
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv("data_eddie/embeddings_PRETRAINED.csv")
    print(f"Loaded dataset with {len(df)} samples")
    print(f"Yield range: {df['Yield'].min():.2f} - {df['Yield'].max():.2f}")

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = EmbeddingDataset(train_df)
    val_dataset = EmbeddingDataset(val_df)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    ligand_dim = len(train_dataset.ligand_cols)
    substrate_dim = len(train_dataset.substrate_cols)

    model = SimpleEmbeddingModel(
        ligand_dim=ligand_dim,
        substrate_dim=substrate_dim,
        hidden_dim=128,  # Smaller model
        tree_depth=3  # Smaller tree
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5
    )

    # Training loop
    num_epochs = 100
    best_val_rmse = float('inf')
    patience_counter = 0
    patience_limit = 15

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            ligand_emb = batch['ligand_emb'].to(device)
            substrate_emb = batch['substrate_emb'].to(device)
            yields = batch['yield'].to(device)

            optimizer.zero_grad()

            predictions = model(ligand_emb, substrate_emb)
            loss = F.mse_loss(predictions, yields)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(yields)
            train_samples += len(yields)

        train_rmse = np.sqrt(train_loss / train_samples)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                ligand_emb = batch['ligand_emb'].to(device)
                substrate_emb = batch['substrate_emb'].to(device)
                yields = batch['yield'].to(device)

                predictions = model(ligand_emb, substrate_emb)
                loss = F.mse_loss(predictions, yields)

                val_loss += loss.item() * len(yields)
                val_samples += len(yields)

        val_rmse = np.sqrt(val_loss / val_samples)

        scheduler.step(val_rmse)

        print(f"Epoch {epoch + 1:3d}/{num_epochs} - "
              f"Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), "best_simple_tree.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Best validation RMSE: {best_val_rmse:.4f}")
    return best_val_rmse


if __name__ == "__main__":
    train_model()