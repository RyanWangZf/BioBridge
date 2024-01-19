"""Build the unified embedding dictionary for all nodes"""
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
emb_path = os.path.join(input_dir, "./embeddings/esm2b_unimo_pubmedbert")

# load data config
with open(os.path.join(output_dir, "data_config.json"), "r") as f:
    data_config = json.load(f)

# load primeKG nodes
df_node = pd.read_csv(os.path.join(primekg_path, "nodes.csv"))

# build a unified embedding matrix for all the raw nodes
# load all embeddings and try to unify them
emb_files = [os.path.join(emb_path, x) for x in os.listdir(emb_path) if x.endswith(".pkl")]

emb_dict_all = {}
emb_dims = {}
for emb_file in tqdm(emb_files):
    print("loading embedding from {}".format(emb_file))
    emb = pickle.load(open(emb_file, "rb"))
    emb_ar = emb["embedding"]
    if not isinstance(emb_ar, list):
        emb_ar = emb_ar.tolist()
    emb_dict = dict(zip(emb["node_index"], emb_ar))
    emb_dict_all.update(emb_dict)

    # save raw embedding dim in data config
    name = os.path.basename(emb_file).replace(".pkl","")

    # map name to original primekg node type
    name_map = {
        "protein": "gene/protein",
        "mf": "molecular_function",
        "cc": "cellular_component",
        "bp": "biological_process",
        "drug": "drug",
        "disease": "disease",
    }

    # transform name to original primekg node type
    name = name_map[name]

    emb_dims[name] = len(emb_ar[0])

data_config["emb_dim"] = emb_dims
# save data config
with open(os.path.join(output_dir, "data_config.json"), "w") as f:
    f.write(json.dumps(data_config, indent=4))

# save
embedding_dict_name = emb_path.split("/")[-1]
f = open(os.path.join(output_dir, "embedding_dict.pkl"), "wb")
pickle.dump(emb_dict_all, f)
f.close()
print("done!")