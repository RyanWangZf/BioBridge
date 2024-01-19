"""Use ESM2 model to encode the protein sequences.
"""
import pdb
from transformers import AutoTokenizer
from transformers import EsmModel
import torch

def load_protein_model(modelname = "facebook/esm2_t6_8M_UR50D"):
    model = EsmModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    return model, tokenizer

def inference(model, batch):
    """average pooling of the last layer of the model considering the different lengths of the sequences in the batch
    ref: https://github.com/facebookresearch/esm
    """
    model.eval()
    with torch.no_grad():
        output = model(**batch)

    attention_mask = batch["attention_mask"]
    emb = output.last_hidden_state # (batch_size, seq_length, hidden_size)
    protein_attention_mask = attention_mask.bool()
    protein_embedding = torch.stack([emb[i,protein_attention_mask[i, :]][1:-1].mean(dim=0) for i in range(len(emb))], dim=0)
    return protein_embedding