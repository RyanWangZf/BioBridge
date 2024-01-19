"""Build configuration of relation types and node types index mapping"""
import os
import pdb
import json
import pandas as pd
from tqdm import tqdm

# data path
input_dir = "./data"
output_dir = "./data/BindData"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
primekg_path = os.path.join(input_dir, "./PrimeKG")

# load primeKG nodes
df_node = pd.read_csv(os.path.join(primekg_path, "nodes.csv"))
nodes = df_node["node_type"].value_counts().index.tolist()
node_vocab = dict(zip(nodes, list(range(len(nodes)))))

# load primeKG triplets
df = pd.read_csv(os.path.join(primekg_path, "kg.csv"))
rels = df["display_relation"].value_counts().index.tolist()
rel_vocab = dict(zip(rels, list(range(len(rels)))))

# build the data config: node type id, relation type id
data_config = {
    "node_type": node_vocab,
    "relation_type": rel_vocab,
}
with open(os.path.join(output_dir, "data_config.json"), "w") as f:
    f.write(json.dumps(data_config, indent=4))

print("done!")