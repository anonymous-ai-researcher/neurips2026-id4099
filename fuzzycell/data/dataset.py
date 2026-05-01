"""PyTorch dataset for single-cell data with CL annotations."""
import torch
from torch.utils.data import Dataset

class CellDataset(Dataset):
    def __init__(self, adata, gene_vocab=None, n_hvg=2000):
        self.adata = adata
        self.gene_vocab = gene_vocab
    def __len__(self):
        return self.adata.n_obs
    def __getitem__(self, idx):
        x = torch.tensor(self.adata.X[idx].toarray().squeeze(), dtype=torch.float32)
        cell_type = self.adata.obs["cell_type"].iloc[idx]
        tissue = self.adata.obs["tissue"].iloc[idx]
        return {"x": x, "cell_type": cell_type, "tissue": tissue}
