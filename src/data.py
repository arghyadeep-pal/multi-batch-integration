# Defining a custom PyTorch Dataset for single-cell data
import torch
from torch.utils.data import Dataset
import numpy as np

class SingleCellDataset(Dataset):
    def __init__(self, adata): # constructor takes in an AnnData object
        # 1. Extract the gene expression matrix from adata.X
        self.gene_matrix = adata.X
        # self.gene_matrix = adata.X.toarray()
        self.cells = adata.obs
        self.genes = adata.var

        # 2. Extract the 'batch' column from adata.obs
        self.batch = adata.obs['batch']

        # 3. Convert the batch strings ('Patient_A', 'Patient_B') into integers (0, 1)
        self.batch = np.where(self.batch == 'Patient_A', 0, 1)
        
        # Note: If adata.X is a sparse matrix, you might need to convert it using .toarray() or 
        # .todense() first!
        pass 

    def __len__(self): # function 1: returns the total number of cells in the dataset
        # Return the total number of cells
        return len(self.cells)
    
        pass

    def __getitem__(self, idx): # function 2
        # 1. Grab the specific cell's gene expression row using the index 'idx'
        get_gene = self.gene_matrix[idx, :] 

        # 2. Grab the corresponding integer batch label
        get_batch = self.batch[idx]

        # 3. Convert both into torch.Tensor objects (use torch.float32 for genes, torch.long for 
        # batches)
        gene_tensor = torch.tensor(get_gene, dtype = torch.float32)
        batch_tensor = torch.tensor(get_batch, dtype = torch.long)

        # 4. Return them as a tuple: (gene_tensor, batch_tensor)
        return (gene_tensor, batch_tensor)

        pass