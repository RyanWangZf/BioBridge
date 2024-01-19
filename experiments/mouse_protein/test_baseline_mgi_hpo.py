"""test the baseline on using mouse protein retrieve human phenotypes.
"""
import os, pickle, pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import defaultdict

from src.text_encoder import load_text_model
from src.text_encoder import inference as text_inference
from src.protein_encoder import load_protein_model
from src.protein_encoder import inference as protein_inference
from src.losses import InfoNCE

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

class DualEncoderModel(nn.Module):
    def __init__(self, protein_encoder, text_encoder):
        super(DualEncoderModel, self).__init__()
        self.loss_fn = InfoNCE()
        self.protein_encoder = protein_encoder
        self.text_encoder = text_encoder

        # freeze protein encoder, only train text encoder
        for param in self.protein_encoder.parameters():
            param.requires_grad = False

        self.protein_proj = nn.Linear(1280, 768, bias=False)
        self.text_proj = nn.Linear(768, 768, bias=False)


    def forward(self, input_protein, input_text, return_loss=True):
        protein_emb = protein_inference(self.protein_encoder, input_protein)
        text_emb = text_inference(self.text_encoder, input_text, enable_grad=True)
        protein_emb = self.protein_proj(protein_emb)
        text_emb = self.text_proj(text_emb)

        if return_loss:
            # compute contrastive loss
            loss = self.loss_fn(protein_emb, text_emb)
            return loss

        return protein_emb, text_emb
    
    def encode_protein(self, input_protein):
        protein_emb = protein_inference(self.protein_encoder, input_protein)
        protein_emb = self.protein_proj(protein_emb)
        return protein_emb
    
    def encode_text(self, input_text):
        text_emb = text_inference(self.text_encoder, input_text)
        text_emb = self.text_proj(text_emb)
        return text_emb

# load data
df_label = pd.read_csv("./data/mouse_protein/mgi_hpo_task/mouse_protein_human_phenotype.csv")
df_mgi = pd.read_csv("./data/mouse_protein/mgi_hpo_task/mgi.csv")
df_hpo = pd.read_csv("./data/mouse_protein/mgi_hpo_task/human_phenotype.csv")
df_hpo["definition"] = df_hpo["definition"].fillna("")
df_hpo["definition"] = "name:" + df_hpo["name"] + " definition:" + df_hpo["definition"]

# load model
checkpoint_path = "./checkpoints/mgi_mpo_baseline/model.pt"
protein_model, protein_tokenizer = load_protein_model("facebook/esm2_t33_650M_UR50D")
text_model, text_tokenizer = load_text_model("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = DualEncoderModel(protein_model, text_model)
model = model.cuda()
protein_tokenizer.model_max_length = 512
text_tokenizer.model_max_length = 512
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# start to encode mgi sequence and hpo definition
if not os.path.exists("./data/mouse_protein/baseline_embedding/mgi_emb.pkl"):
    print("start to encode mgi sequence and hpo definition")
    # tokenize all mgi sequence
    protein_inputs = [protein_tokenizer(v, truncation=True) for v in df_mgi["sequence"]]
    protein_emb_all = []
    bs = 16
    for idx in trange(0, len(protein_inputs), bs):
        protein_inputs_batch = protein_inputs[idx:idx+bs]
        protein_inputs_batch = protein_tokenizer.pad(protein_inputs_batch, return_tensors="pt")
        protein_inputs_batch = {k: v.cuda() for k, v in protein_inputs_batch.items()}
        with torch.no_grad():
            protein_emb = model.encode_protein(protein_inputs_batch)
        protein_emb = protein_emb.cpu().detach().numpy()
        protein_emb_all.append(protein_emb)
    protein_emb_all = np.concatenate(protein_emb_all, axis=0)
    with open("./data/mouse_protein/baseline_embedding/mgi_emb.pkl", "wb") as f:
        to_save = {
            "embedding": protein_emb_all,
            "mgi": df_mgi["MGI"].values
        }
        pickle.dump(to_save, f)

if not os.path.exists("./data/mouse_protein/baseline_embedding/hpo_emb.pkl"):
    # tokenize all hpo definition
    text_inputs = [text_tokenizer(v, truncation=True) for v in df_hpo["definition"].tolist()]
    phenotype_emb_all = []
    bs = 16
    for idx in trange(0, len(text_inputs), bs):
        text_inputs_batch = text_inputs[idx:idx+bs]
        text_inputs_batch = text_tokenizer.pad(text_inputs_batch, return_tensors="pt")
        text_inputs_batch = {k: v.cuda() for k, v in text_inputs_batch.items()}
        with torch.no_grad():
            phenotype_emb = model.encode_text(text_inputs_batch)
        phenotype_emb = phenotype_emb.cpu().detach().numpy()
        phenotype_emb_all.append(phenotype_emb)
    phenotype_emb_all = np.concatenate(phenotype_emb_all, axis=0)
    with open("./data/mouse_protein/baseline_embedding/hpo_emb.pkl", "wb") as f:
        to_save = {
            "embedding": phenotype_emb_all,
            "hpo": df_hpo["HPO"].values
        }
        pickle.dump(to_save, f)

# load embeddings
mgi_embs = pickle.load(open("./data/mouse_protein/baseline_embedding/mgi_emb.pkl", "rb"))
hpo_embs = pickle.load(open("./data/mouse_protein/baseline_embedding/hpo_emb.pkl", "rb"))

# compute pairwise similarity
test_mgi_emb = mgi_embs["embedding"]
test_mgi_emb = test_mgi_emb / np.linalg.norm(test_mgi_emb, axis=1, keepdims=True)
test_hpo_emb = hpo_embs["embedding"]
test_hpo_emb = test_hpo_emb / np.linalg.norm(test_hpo_emb, axis=1, keepdims=True)
cos_sim = np.matmul(test_mgi_emb, test_hpo_emb.T)

# get label
results = defaultdict(list)
for idx, row in df_label.iterrows():
    mgi = row["MGI"]
    gt = row["human_phenotype_id"]
    gt = eval(gt)

    # get the position
    pred = cos_sim[idx].argsort()[::-1]
    pred = hpo_embs["hpo"][pred]
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