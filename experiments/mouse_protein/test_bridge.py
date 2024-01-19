"""Load pre-trained bridge model, project the data and make retrieval task.
"""
import os, pickle, pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from src.model import BindingModel

checkpoint_dir = "./checkpoints/bind-openke-benchmark-6-layer-unimol"

# load the data
input_dir = "./data/mouse_protein/"
df_test = pd.read_csv(os.path.join(input_dir, "processed/train_test_split/triplet_test.csv"))
df_test_mgi = pd.read_csv(os.path.join(input_dir, "processed/train_test_split/test_mgi.csv"))
df_test_mpi = pd.read_csv(os.path.join(input_dir, "processed/train_test_split/test_mpi.csv"))

# get the protein and phenotype embeddings
label = df_test.groupby("MGI").agg({"MPI": lambda x: list(x)}).reset_index()
label.set_index("MGI", inplace=True)

# load the raw protein and phenotype embeddings waiting for projection
input_emb_dir = os.path.join(input_dir, "embeddings")
with open(os.path.join(input_emb_dir, "mgi.pkl"), "rb") as f:
    pro_emb = pickle.load(f)

with open(os.path.join(input_emb_dir, "mpi.pkl"), "rb") as f:
    mpi_emb = pickle.load(f)

# filter pro and mpi embeddings, only keep the ones in the test set
pro_index = np.array(pro_emb["mgi"])
pro_emb = pro_emb["embedding"]
pro_index = pd.Series(pro_index, name="MGI")
pro_index = pro_index[pro_index.isin(label.index)]
pro_index = pro_index.drop_duplicates()
pro_emb = pro_emb[pro_index.index]
label = label.loc[pro_index]

# get MPI embeddings
mpi_index = np.array(mpi_emb["mpi"])
mpi_emb = mpi_emb["embedding"]
mpi_index = pd.Series(mpi_index, name="MPI")
mpi_index = mpi_index[mpi_index.isin(df_test_mpi["MPI"])]
mpi_index = mpi_index.drop_duplicates()
mpi_emb = mpi_emb[mpi_index.index]

# load model
print("### Model Configuration ###")
with open(os.path.join(checkpoint_dir, "model_config.json"), "r") as f:
    model_config = json.load(f)
model = BindingModel(**model_config)
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin")))
model.cuda()
model.eval()

print("### Data Config ###")
with open("/home/ec2-user/data/BindData/data_config.json", "r") as f:
    data_config = json.load(f)

# try protein - associated with - disease
head_type_id = data_config["node_type"]["gene/protein"]
# tail_type_id = data_config["node_type"]["disease"]
# rel_type_id = data_config["relation_type"]["associated with"]

# try protein - interacts with - bp
tail_type_id = data_config["node_type"]["biological_process"]
rel_type_id = data_config["relation_type"]["interacts with"]

# try protein - interacts with - mf
# tail_type_id = data_config["node_type"]["molecular_function"]
# rel_type_id = data_config["relation_type"]["interacts with"]

# try protein - interacts with - cc
# tail_type_id = data_config["node_type"]["cellular_component"]
# rel_type_id = data_config["relation_type"]["interacts with"]

# # try protein - interacts with - disease
# tail_type_id = data_config["node_type"]["disease"]
# rel_type_id = data_config["relation_type"]["interacts with"]

mpi_emb = torch.tensor(mpi_emb, dtype=torch.float32).cuda()
with torch.no_grad():
    mpi_tgt_emb = model.projection(mpi_emb, tail_type_id).cpu().numpy()

# encode the protein embeddings
pro_emb = torch.tensor(pro_emb, dtype=torch.float32).cuda()
pro_tgt_emb = model.encode(pro_emb, head_type_id, rel_type_id, tail_type_id, batch_size=1024)

# compute the similarity
test_mgi_emb = pro_tgt_emb / (pro_tgt_emb ** 2).sum(axis=1, keepdims=True) ** 0.5
test_mpi_emb = mpi_tgt_emb / (mpi_tgt_emb ** 2).sum(axis=1, keepdims=True) ** 0.5
cos_sim = test_mgi_emb @ test_mpi_emb.T

label = label.reset_index()
results = defaultdict(list)
for idx, row in tqdm(label.iterrows(), total=len(label)):
    pred = cos_sim[idx].argsort()[::-1]
    pred = mpi_index.values[pred]
    gt = row["MPI"]

    for k in [1, 3, 5]:
        # compute recall@k
        results["recall@{}".format(k)].append(len(set(pred[:k]) & set(gt)) / len(gt))
        # compute precision@k
        results["precision@{}".format(k)].append(len(set(pred[:k]) & set(gt)) / k)
        # compute ndcg@k
        dcg = 0
        for i, p in enumerate(pred[:k]):
            if p in gt:
                dcg += 1 / np.log2(i + 2)
        idcg = 0
        for i in range(min(k, len(gt))):
            idcg += 1 / np.log2(i + 2)
        results["ndcg@{}".format(k)].append(dcg / idcg)


for k, v in results.items():
    print("{}: {}".format(k, np.mean(v)))

print("done!")

pdb.set_trace()