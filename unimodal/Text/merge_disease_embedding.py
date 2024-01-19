"""
The raw disease node index may map to multiple diseases (node_id), so we need to take a mean pooling of those disease embeddings
for a node.
"""
import os
import pandas as pd
import pickle
import pdb
input_dir = "./data/embeddings"

with open(os.path.join(input_dir, "disease.pkl"), "rb") as f:
    embs = pickle.load(f)

# move the raw embeddings to a new file
with open(os.path.join(input_dir, "./raw_embeddings/disease_not_grouped.pkl"), "wb") as f:
    pickle.dump(embs, f)

df = pd.DataFrame({
        "node_index": embs["node_index"],
        }
    )
df = df.reset_index()

grouped = df.groupby("node_index").agg(list).reset_index()

emb_list = []
for idx, row in grouped.iterrows():
    emb = embs["embedding"][row["index"]]
    emb = emb.mean(0)
    emb_list.append(emb)
new_outputs = {
    "node_index": grouped["node_index"].tolist(),
    "embedding": emb_list,
}

# save grouped disease embeddings
with open(os.path.join(input_dir, "disease.pkl"), "wb") as f:
    pickle.dump(new_outputs, f)

print("done!")