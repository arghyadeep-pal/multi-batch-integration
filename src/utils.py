"""
src/utils.py — shared project foundation.

Three responsibilities:
  1. Reproducibility — one call that seeds every RNG the project touches.
  2. Canonical paths — repo-root-relative locations, created on import, so no
     notebook ever hardcodes an absolute path.
  3. Embedding persistence — save/load helpers that enforce the project's
     embedding contract: every arm produces an (n_cells, d) array whose rows
     align to the canonical AnnData's obs_names. These aren't exercised until
     notebook 01, but they live here so the module is complete.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 1. Reproducibility
# --------------------------------------------------------------------------- #
def set_seeds(seed: int = 0) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA) from a single value.

    Call at the top of every notebook, right after imports, so a run is
    reproducible regardless of which arm executes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)   # affects subprocesses / hash order
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Low-risk determinism nudge for cuDNN; safe to keep on at this scale.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# 2. Canonical paths
# --------------------------------------------------------------------------- #
# This file lives in src/, so the repo root is two levels up. Everything else is
# defined relative to it — that's why the notebooks stay path-agnostic.
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW       = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS        = REPO_ROOT / "results"
EMBEDDINGS     = REPO_ROOT / "results" / "embeddings"
FIGURES        = REPO_ROOT / "figures"

# Create any that don't exist yet. Idempotent and safe to run on import — the
# same habit as the os.makedirs(..., exist_ok=True) calls in notebook 01.
for _p in (DATA_RAW, DATA_PROCESSED, RESULTS, EMBEDDINGS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 3. Embedding persistence  (used from notebook 01 onward)
# --------------------------------------------------------------------------- #
# The contract: an embedding is an (n_cells, d) float array whose row order
# matches the canonical AnnData's obs_names. We persist the obs_names alongside
# the array so alignment can be *checked*, not assumed — the single most likely
# silent failure in this project is an embedding (e.g. computed on Kaggle)
# coming back in a different row order and being scored against the wrong cells.

def save_embedding(name: str, array: np.ndarray, obs_names) -> Path:
    """Persist one arm's embedding to results/embeddings/<name>.npz.

    Stores the (n_cells, d) array together with the cell identifiers it was
    computed on. Asserts the row count matches the cell count before writing.

    Parameters
    ----------
    name : short arm key, e.g. "scvi", "scgpt", "myvae".
    array : (n_cells, d) NumPy array. If you have a torch tensor, pass
        tensor.cpu().numpy().
    obs_names : the AnnData's obs_names the embedding was computed on
        (adata.obs_names).
    """
    array = np.asarray(array)
    obs_names = np.asarray(obs_names, dtype=str)
    if array.shape[0] != obs_names.shape[0]:
        raise ValueError(
            f"'{name}': {array.shape[0]} embedding rows but "
            f"{obs_names.shape[0]} cell names — row/cell count mismatch."
        )
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    path = EMBEDDINGS / f"{name}.npz"
    np.savez(str(path), array=array, obs_names=obs_names)
    return path


def load_embedding(name: str):
    """Load an arm's embedding back as (array, obs_names).

    The caller verifies order against the current AnnData before attaching —
    that order check is the other half of the contract. In notebook 04:

        emb, names = load_embedding("scvi")
        assert np.array_equal(names, adata.obs_names.to_numpy().astype(str)), \\
            "scvi embedding is not row-aligned to the canonical AnnData"
        adata.obsm["X_scvi"] = emb
    """
    path = EMBEDDINGS / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No saved embedding at {path}")
    data = np.load(str(path), allow_pickle=False)
    return data["array"], data["obs_names"]