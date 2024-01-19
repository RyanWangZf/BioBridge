"""Based on the extracted triplets, build the train/test split for the following experiments.

TODO: hold out for downstream tasks?s
"""
import os
import pdb
import json
import pickle
import pandas as pd
import numpy as np
np.random.seed(42)
from tqdm import tqdm

# data path
input_dir = "./data"
output_dir = "./data/BindData/train_test_split"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
primekg_path = os.path.join(input_dir, "./PrimeKG")
df_kg = pd.read_csv(os.path.join(primekg_path, "kg.csv"))

# load data config
with open(os.path.join(input_dir, "./BindData/data_config.json"), "r") as f:
    data_config = json.load(f)
df_full = pd.read_csv(os.path.join(input_dir, "./BindData/triplet_full.csv"))

# hold out by separating head node indexes
train_list, test_list = [], []
tr_node_split = {
    "node_index": [],
    "node_type": [],
}
te_node_split = {
    "node_index": [],
    "node_type": [],
}

for node_type in df_full["x_type"].unique():
    # hold out by separating head node indexes
    # the test set has 10% of all the nodes
    df_sub = df_full[df_full["x_type"] == node_type]
    all_x_indexes = df_sub["x_index"].unique()
    
    # sample 10% from all_x_indexes, and hold out as the test set
    te_x_indexes = np.random.choice(all_x_indexes, size=int(0.1*len(all_x_indexes)), replace=False)
    df_sub_te = df_sub[df_sub["x_index"].isin(te_x_indexes)]
    df_sub_tr = df_sub[~df_sub["x_index"].isin(te_x_indexes)]
    train_list.append(df_sub_tr)
    test_list.append(df_sub_te)
    
    # record the split
    tr_node_index = df_sub_tr["x_index"].unique()
    te_node_index = df_sub_te["x_index"].unique()
    tr_node_split["node_index"].extend(tr_node_index.tolist())
    tr_node_split["node_type"].extend([node_type]*len(tr_node_index))

    te_node_split["node_index"].extend(te_node_index.tolist())
    te_node_split["node_type"].extend([node_type]*len(te_node_index))

    print("number of {} nodes in train: {}".format(node_type, len(tr_node_index)))
    print("number of {} nodes in test: {}".format(node_type, len(te_node_index)))

# concatenate all dataframes
df_train = pd.concat(train_list)
df_test = pd.concat(test_list)

# save train/test split
df_train.to_csv(os.path.join(output_dir, "triplet_train.csv"), index=False)
df_test.to_csv(os.path.join(output_dir, "triplet_test.csv"), index=False)

df_node_train = pd.DataFrame(tr_node_split)
df_node_test = pd.DataFrame(te_node_split)
df_node_train.to_csv(os.path.join(output_dir, "node_train.csv"), index=False)
df_node_test.to_csv(os.path.join(output_dir, "node_test.csv"), index=False)

print("done")
