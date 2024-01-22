# Run this script to build the data for the following experiments

## 0. Prepare for data processing
Please download the raw PrimeKG data from: [PrimeKG](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM).
Then, unzip the primekg data and put it under the data folder
```shell
unzip data/PrimeKG/dataverse_files.zip -d data/PrimeKG
```

The preprocessed node features are under the folder `data/Processed/`.

The pre-encoded node embeddings are under the folder `data/embeddings`.

## 1. build the data configurations

Attention: since the `data_config.json` used by our experiments is already provided in this repo at `./data/BindData`, you can **skip** this step if you just need to run the pre-trained models and load the `data_config.json`.

Run this step will generate the `data_config.json` under the folder `data/BindData/`.
And you will need to train the whole model from scratch.

```shell
python build_data_config.py
```

## 2. build the node embedding dictionary
```shell
python build_node_emb.py
```
Attention:
In our experiments, UniMol fails to encode the following drug nodes
```
node_index = [14186, 14736, 14737, 19929, 19948, 20103, 20271, 20832]
drugbank_id = ['DB00515', 'DB00526', 'DB00958', 'DB08276', 'DB01929', 'DB04156', 'DB04100', 'DB13145']
```
so `embedding_dict.pkl` won't have these nodes and we will drop these nodes from the training dataset in step \# 4.

It is encouraged to investigate the reason why UniMol fails to encode these nodes and add the encoded embeddings to `embedding_dict.pkl`.

UniMol representation can be found in the guidance: https://github.com/dptech-corp/Uni-Mol/tree/main/unimol_tools#unimol-molecule-and-atoms-level-representation 

## 3. build the full triplet set by filtering out from PrimeKG
```shell
python build_full_triplets.py
```

## 4. build the train/test split
```shell
python build_train_test_split.py
```

## 5. build the negative triplets for the training split
```shell
python build_negative_samples.py --split_folder "./data/BindData/train_test_split" 
```