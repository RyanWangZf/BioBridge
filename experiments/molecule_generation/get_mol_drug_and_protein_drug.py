# test on drugbank dataset,
# taking disease or sth as input

import pdb
import os
import json
import pickle
import torch
import pandas as pd

from src.model import BindingModel
from src.inference import BridgeInference

# prompt for generating drug molecules
# prompt for generation protein-based therapies 
# df = pd.read_csv("./data/DrugBank/drugbank.csv")
# get protein-based therapies
# protein_df = df[~df["protein_formula"].isnull()]
output_dir = "./data/generation_data"

# get the data config
with open(os.path.join("./data/BindData/", "data_config.json"), "r") as f:
    data_config = json.load(f)

# retrieve disease using molecule / protein
checkpoint_dir = "./checkpoints/bind-openke-benchmark-6-layer-unimol"
with open(os.path.join(checkpoint_dir, "model_config.json"), "r") as f:
    model_config = json.load(f)
model = BindingModel(**model_config)
model = BridgeInference(model)
mol_raw_dir = "./data/embeddings/esm2b_unimo_pubmedbert/drug.pkl"
with open(mol_raw_dir, "rb") as f:
    mol_raw = pickle.load(f)
pro_raw_dir = "./data/embeddings/esm2b_unimo_pubmedbert/protein.pkl"
with open(pro_raw_dir, "rb") as f:
    pro_raw = pickle.load(f)

# transform molecule to the disease space
# res = model.transform(
#     x = torch.tensor(mol_raw['embedding'], dtype=torch.float32),
#     src_type = 6, # mol 1: protein
#     tgt_type = 2, # disease
#     rel_type = 11, # indication
# )

# transform protein to disease space

# project disease from text embeddings to the disease space
dis_raw_dir = "./data/embeddings/esm2b_unimo_pubmedbert/disease.pkl"
with open(dis_raw_dir, "rb") as f:
    dis_raw = pickle.load(f)
res = model.project(
    x = torch.tensor(dis_raw['embedding'], dtype=torch.float32),
    src_type = 2, # disease
)