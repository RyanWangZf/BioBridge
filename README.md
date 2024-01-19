# BioBridge

This is the official implementation of the paper [BioBridge: Bridging Biomedical Foundation Models via Knowledge Graph](https://arxiv.org/pdf/2310.03320.pdf) (ICLR 2024).

# Main components

## `checkpoints/`
The trained BioBridge model checkpoints.


## `data/`
- `data/PrimeKG`: The raw PrimeKG data in `.zip` format.

- `data/Processed`: The node features obtained from different databases, e.g., protein's sequence.

- `data/embeddings/`: The KG node embeddings extracted from unimodal FMs, such as PubMedBERT, UniMol, and ESM-2.

- `data/BindData/`: The preprocessed BioBridge related data and its configurations.

## `dataset/`
The guidelines for data preprocessing.

## `src/`
The source code of BioBridge.

## `unimodal/`
The source code of unimodal FMs, including PubMedBERT and ESM-2, for encoding node features.

## `train.sh`
The example script for training BioBridge.

## `eval.sh`
The example script for using BioBridge for cross-modality prediction.



