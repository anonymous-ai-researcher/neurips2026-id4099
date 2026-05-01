"""Gene expression preprocessing."""
import scanpy as sc

def preprocess_counts(adata, target_sum=10000, n_hvg=2000):
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    return adata[:, adata.var["highly_variable"]]
