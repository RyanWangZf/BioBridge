"""Load LMs to encode text modalities in the KG.
"""
import pdb
from transformers import AutoModel, AutoTokenizer
import torch

def load_text_model(modelname = "dmis-lab/biobert-base-cased-v1.2"):
    model = AutoModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    return model, tokenizer

def inference(model, batch, enable_grad=False):
    if enable_grad:
        output = model(**batch)
    else:
        with torch.no_grad():
            output = model(**batch)

    emb = output.last_hidden_state
    attention_mask = batch["attention_mask"]
    seq_len = batch["attention_mask"].sum(dim=1)
    emb = emb * attention_mask.unsqueeze(-1)
    emb = emb.sum(dim=1) / seq_len.unsqueeze(-1)
    return emb