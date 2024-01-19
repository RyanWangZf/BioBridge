"""load the baseline model and encode the data.
"""

"""test the baseline embeddings for the mouse protein phenotype retrieval task.
"""
import os, pickle, pdb
import pandas as pd
from collections import defaultdict
import numpy as np
from tqdm import tqdm

input_dir = "./data/mouse_protein/"

# load the data
# pro_emb_file = os.path.join(input_dir, "./baseline_embedding/mouse_protein.pkl")
# mpi_emb_file = os.path.join(input_dir, "./baseline_embedding/mouse_phenotype.pkl")

pro_emb_file = os.path.join(input_dir, "./baseline_embedding/protein_emb.pkl")
mpi_emb_file = os.path.join(input_dir, "./baseline_embedding/phenotype_emb.pkl")


with open(pro_emb_file, "rb") as f:
    pro_emb = pickle.load(f)

with open(mpi_emb_file, "rb") as f:
    mpi_emb = pickle.load(f)

# load the data
df_test = pd.read_csv(os.path.join(input_dir, "processed/train_test_split/triplet_test.csv"))
df_test_mgi = pd.read_csv(os.path.join(input_dir, "processed/train_test_split/test_mgi.csv"))
df_test_mpi = pd.read_csv(os.path.join(input_dir, "processed/train_test_split/test_mpi.csv"))

# get the protein and phenotype embeddings
all_test_mgi_index = df_test.drop_duplicates("MGI")
test_mgi_emb = pro_emb[all_test_mgi_index.index]
test_mgi_index = all_test_mgi_index["MGI"].values

all_test_mpi_index = df_test.drop_duplicates("MPI")
test_mpi_emb = mpi_emb[all_test_mpi_index.index]
test_mpi_index = all_test_mpi_index["MPI"].values

label = df_test.groupby("MGI").agg({"MPI": lambda x: list(x)}).reset_index()
label.set_index("MGI", inplace=True)
label = label.loc[test_mgi_index]

# start to go through the test set protein and compute the similarity
# between the protein and all the phenotypes
test_mgi_emb = test_mgi_emb / (test_mgi_emb ** 2).sum(axis=1, keepdims=True) ** 0.5
test_mpi_emb = test_mpi_emb / (test_mpi_emb ** 2).sum(axis=1, keepdims=True) ** 0.5
cos_sim = test_mgi_emb @ test_mpi_emb.T

label = label.reset_index()
results = defaultdict(list)
for idx, row in tqdm(label.iterrows(), total=len(label)):
    pred = cos_sim[idx].argsort()[::-1]
    pred = test_mpi_index[pred]
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
