import pdb
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Old implementation with MolT5
# def load_molecule_model(modelname = "laituan245/molt5-large-smiles2caption"):
#     model = AutoModel.from_pretrained(modelname)
#     tokenizer = AutoTokenizer.from_pretrained(modelname)
#     return model, tokenizer

# def inference(model, batch):
#     with torch.no_grad():
#         output = model.get_encoder()(**batch)
    
#     attention_mask = batch["attention_mask"]
#     seq_len = attention_mask.sum(dim=1)
#     emb = output["last_hidden_state"]

#     # average pooling
#     res = (emb * attention_mask.unsqueeze(-1)).sum(dim=1) / seq_len.unsqueeze(-1)
#     return res


def load_molecule_model(modelname = None):
    from unimol_tools import UniMolRepr
    model = UniMolRepr(data_type="molecule")
    return model, None

def inference(model, batch):
    """encode SMILES strings with UniMolRepr

    Args:
        model (UniMolRepr): the UniMolRepr model
        batch (list): a list of SMILES strings
    """
    reprs = model.get_repr(batch)
    (
        # dict_keys(['cls_repr', 'atomic_reprs'])
        reprs.keys(),  
        # torch.Size([3, 512])
        torch.tensor(reprs["cls_repr"]).shape,  
        # [torch.Size([9, 512]), torch.Size([11, 512]), torch.Size([14, 512])])
        [torch.tensor(x).shape for x in reprs["atomic_reprs"]]  
    )
    emb = np.array(reprs["cls_repr"])
    return emb