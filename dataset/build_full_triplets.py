"""Build the train / test split for the training data. 
Each sample is a dict object containing the following keys
{
    "head_id": "xxx", # node index
    "tail_id": "xxx", # node index
    "head_type_id": "xxx", # node type id
    "tail_type_id": "xxx", # node type id
    "relation_id": "xxx", # relation type id
    "neg_tail_id": ["xx","xx","xxx","xxxx", "xxxx"], # sampled from the same node type as tail but has no relations with head.
}
"""
import os
import pdb
import json
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# data path
input_dir = "./data"
output_dir = "./data/BindData"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
primekg_path = os.path.join(input_dir, "./PrimeKG")
node_path = os.path.join(input_dir, "./Processed")
df_kg = pd.read_csv(os.path.join(primekg_path, "kg.csv"))

# load data config
with open(os.path.join(output_dir, "data_config.json"), "r") as f:
    data_config = json.load(f)

# the current modalities that we want to involve
node_file_list = [
    "biological.csv",
    "disease.csv",
    "cellular.csv",
    "molecular.csv",
    "protein.csv",
    "drug.csv",
]

node_index_list = []
for node_file in node_file_list:
    df_node = pd.read_csv(os.path.join(node_path, node_file))
    node_index_list.extend(df_node["node_index"].tolist())

# only keep the triplets that have nodes in the six modalities
df_kg_sub = df_kg[df_kg["x_index"].isin(node_index_list) & df_kg["y_index"].isin(node_index_list)]
df_kg_sub = df_kg_sub.reset_index(drop=True)

df_kg_sub = df_kg_sub[["x_index", "y_index", "display_relation", "relation", "x_type", "y_type"]]

# map node type to index according to data config
df_kg_sub["x_type"] = df_kg_sub["x_type"].apply(lambda x: data_config["node_type"][x])
df_kg_sub["y_type"] = df_kg_sub["y_type"].apply(lambda x: data_config["node_type"][x])

# map relation type to index according to data config
df_kg_sub["display_relation"] = df_kg_sub["display_relation"].apply(lambda x: data_config["relation_type"][x])

# save the triplets
df_kg_sub.to_csv(os.path.join(output_dir, "triplet_full.csv"), index=False)

print("done!")