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

- `data/mouse_protein/`: The preprocessed mouse protein data and its configurations regarding the mouse protein prediction task in the paper.

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

## `experiments/mouse_protein`
The example script for training BioBridge on the mouse protein prediction task.

## `experiments/molecule_generation`
The example script for testing BioBridge on prompting LLMs for molecule generation and Q&A tasks.

# References
If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{wang2023biobridge,
  title={BioBridge: Bridging Biomedical Foundation Models via Knowledge Graph},
  author={Wang, Zifeng and Wang, Zichen and Srinivasan, Balasubramaniam and Ioannidis, Vassilis N and Rangwala, Huzefa and Anubhai, Rishita},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

