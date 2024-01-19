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

```shell
python build_data_config.py
```

## 2. build the node embedding dictionary
```shell
python build_node_emb.py
```

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