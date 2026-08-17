# Single-Protocol Benchmark of scRNA-seq Batch Integration Methods

Comparing classical, purpose-built, and foundation-model approaches to single-cell
batch integration under one common evaluation protocol on one dataset.

**Status: in progress.** Currently at data preparation (notebook 00). No integration
results yet.

---

## Why

Methods that claim to remove technical batch effects while preserving biological
signal each publish their own evaluation — on their own dataset, with their own
preprocessing, and their own choice of metrics. Those numbers are not comparable
to each other.

This repository runs several method families through a single protocol on a single
dataset, so that they are.

## Dataset

Human pancreas benchmark from [scIB](https://github.com/theislab/scib)
([figshare](https://figshare.com/ndownloader/files/24539828)).

| | |
|---|---|
| Cells | 16,382 |
| Genes | 19,093 |
| Batches | 9 sequencing technologies |
| Labels | 14 curated cell types |

Nine protocols spanning droplet (inDrop 1–4), plate-based full-length
(smartseq2, smarter, fluidigmc1), and CEL-Seq (celseq, celseq2) chemistries —
severe enough batch structure to be worth measuring.

## Arms

| Arm | Family | Status |
|---|---|---|
| PCA (uncorrected) | baseline | not started |
| Harmony | classical | not started |
| scVI | purpose-built generative | not started |
| scANVI | purpose-built generative (label-aware) | not started |
| scGPT | foundation model, zero-shot | not started |
| Geneformer | foundation model, zero-shot | not started |
| Conditional VAE | own implementation | designed, not written |

All arms are scored with [`scib-metrics`](https://github.com/YosefLab/scib-metrics)
on both batch correction and biological conservation. The two trade off against
each other; a method that wins on one alone has not been shown to work.

## The embedding contract

Every arm must emit an `(n_cells, d)` array, persisted together with the
`obs_names` it was computed on.

```python
save_embedding("scvi", Z, adata.obs_names)   # refuses to write on a row/cell mismatch
emb, names = load_embedding("scvi")          # returns both, so order can be checked
assert np.array_equal(names, adata.obs_names.to_numpy().astype(str))
```

This exists because the most likely silent failure in a project like this is an
embedding computed elsewhere — a different machine, a different session — coming
back in a different row order and being scored against the wrong cells. That
produces a plausible number, not an error.

`src/utils.py` also seeds Python, NumPy and Torch from a single call, so a run
does not depend on which arm executed first.

## Layout

```
data/raw/           downloaded source data (gitignored)
data/processed/     canonical AnnData object
notebooks/          00_data_prep, 01+ per-arm embeddings, scoring
results/embeddings/ one .npz per arm
src/utils.py        seeding, paths, embedding persistence
```

## Findings so far (notebook 00)

- **`X` is log-normalized, not raw.** Established by comparing against
  `layers['counts']` rather than inferring from the field name.
- **Gene identifiers are HGNC symbols**, so Ensembl mapping is required before
  Geneformer tokenization.
- **The `counts` layer is not uniformly integer.** Checking protocol by protocol,
  four of the nine technologies (celseq, celseq2, fluidigmc1, smarter) carry
  non-integer values. This constrains which likelihood the count-based models can
  legitimately use, and is not something the schema announces.

## Constraints

Run on consumer hardware and free-tier GPU sessions. Where an arm doesn't fit in
available memory, the workaround is documented in that arm's notebook rather than
silently changing the protocol.

## Reproducing

```bash
pip install -r requirements.txt
# notebooks run in order; 00 must complete before any other
```

Seed is fixed at 0 throughout (`src.utils.set_seeds`).

## License

MIT.
