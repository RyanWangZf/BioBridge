"""Build the train/test split for the mouse protein dataset.
"""

import os
import pdb
import pandas as pd
import numpy as np
np.random.seed(42)

input_folder = "/home/ec2-user/data/mouse_protein/processed"

df_hpp = pd.read_csv(os.path.join(input_folder, "human_protein_phenotype.csv"))
df_mgi = pd.read_csv(os.path.join(input_folder, "MGI_MPI_sequence.csv"))
df_mp = pd.read_csv(os.path.join(input_folder, "MP.csv"))

df_mgi["MPI"] = df_mgi["MPI"].apply(lambda x: eval(x))

df_mgi = df_mgi.explode("MPI")
df_full = df_mgi.merge(df_mp, on="MPI").reset_index(drop=True)

df_full.to_csv(os.path.join(input_folder, "triplet_full.csv"), index=False)

# Split the dataset into train/test
# 50% train, 50% test, split by MGI
df_mgi = df_full["MGI"].unique()
np.random.shuffle(df_mgi)
df_mgi_train = df_mgi[:int(len(df_mgi)/2)]
df_mgi_test = df_mgi[int(len(df_mgi)/2):]
df_tr = df_full[df_full["MGI"].isin(df_mgi_train)].reset_index(drop=True)
df_te = df_full[df_full["MGI"].isin(df_mgi_test)].reset_index(drop=True)

df_tr.to_csv(os.path.join(input_folder, "train_test_split/triplet_train.csv"), index=False)
df_te.to_csv(os.path.join(input_folder, "train_test_split/triplet_test.csv"), index=False)

# get train MGI, test MGI, train MPI, test MPI
train_mgi = df_tr["MGI"].unique()
test_mgi = df_te["MGI"].unique()
train_mpi = df_tr["MPI"].unique()
test_mpi = df_te["MPI"].unique()

train_mgi = pd.DataFrame(train_mgi, columns=["MGI"]).to_csv(os.path.join(input_folder, "train_test_split/train_mgi.csv"), index=False)
test_mgi = pd.DataFrame(test_mgi, columns=["MGI"]).to_csv(os.path.join(input_folder, "train_test_split/test_mgi.csv"), index=False)
train_mpi = pd.DataFrame(train_mpi, columns=["MPI"]).to_csv(os.path.join(input_folder, "train_test_split/train_mpi.csv"), index=False)
test_mpi = pd.DataFrame(test_mpi, columns=["MPI"]).to_csv(os.path.join(input_folder, "train_test_split/test_mpi.csv"), index=False)

print("done!")



