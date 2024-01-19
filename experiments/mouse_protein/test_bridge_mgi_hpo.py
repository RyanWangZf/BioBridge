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
df_hpo = pd.read_csv(os.path.join(input_dir, "mgi_hpo_task/human_phenotype.csv"))
df_mgi = pd.read_csv(os.path.join(input_dir, "mgi_hpo_task/mgi.csv"))
df_label = pd.read_csv(os.path.join(input_dir, "mgi_hpo_task/mouse_protein_human_phenotype.csv"))
df_label.set_index("MGI", inplace=True)

# load the raw protein and phenotype embeddings waiting for projection
input_emb_dir = os.path.join(input_dir, "embeddings")
with open(os.path.join(input_emb_dir, "mgi_for_hpo.pkl"), "rb") as f:
    pro_emb = pickle.load(f)

with open(os.path.join(input_emb_dir, "hpo.pkl"), "rb") as f:
    hpo_emb = pickle.load(f)

pro_index = np.array(pro_emb["mgi"])
pro_emb = pro_emb["embedding"]
pro_index = pd.Series(pro_index, name="MGI")
label = df_label.loc[pro_index]
hpo_index = np.array(hpo_emb["HPO"])
hpo_emb = hpo_emb["embedding"]

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
tail_type_id = data_config["node_type"]["disease"]
rel_type_id = data_config["relation_type"]["associated with"]

# try protein - interacts with - bp
# tail_type_id = data_config["node_type"]["biological_process"]
# rel_type_id = data_config["relation_type"]["interacts with"]

# try protein - interacts with - mf
# tail_type_id = data_config["node_type"]["molecular_function"]
# rel_type_id = data_config["relation_type"]["interacts with"]

# try protein - interacts with - cc
# tail_type_id = data_config["node_type"]["cellular_component"]
# rel_type_id = data_config["relation_type"]["interacts with"]

# # try protein - interacts with - disease
# tail_type_id = data_config["node_type"]["disease"]
# rel_type_id = data_config["relation_type"]["interacts with"]

hpo_emb = torch.tensor(hpo_emb, dtype=torch.float32).cuda()
with torch.no_grad():
    hpo_tgt_emb = model.projection(hpo_emb, tail_type_id).cpu().numpy()

# encode the protein embeddings
pro_emb = torch.tensor(pro_emb, dtype=torch.float32).cuda()
pro_tgt_emb = model.encode(pro_emb, head_type_id, rel_type_id, tail_type_id, batch_size=1024)

# compute the similarity
test_mgi_emb = pro_tgt_emb / (pro_tgt_emb ** 2).sum(axis=1, keepdims=True) ** 0.5
test_hpo_emb = hpo_tgt_emb / (hpo_tgt_emb ** 2).sum(axis=1, keepdims=True) ** 0.5
cos_sim = test_mgi_emb @ test_hpo_emb.T

label = label.reset_index()
results = defaultdict(list)
for idx, row in tqdm(label.iterrows(), total=len(label)):
    pred = cos_sim[idx].argsort()[::-1]
    gt = eval(row["human_phenotype_id"])
    pred = hpo_index[pred]

    for k in [1, 5, 10, 20]:
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