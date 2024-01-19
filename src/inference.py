"""Inference based on the trained Bridge model to project raw embeddings to the target space.
"""
import pdb

import torch
import numpy as np

from .model import BindingModel

@torch.no_grad()
def transformation(
    model: BindingModel, 
    x: torch.Tensor,
    src_type,
    tgt_type,
    rel_type
    ):
    """Inference based on the trained Bridge model to project raw embeddings to the target space.

    Args:
        model (BindingModel): the trained Bridge model.
        x (torch.Tensor): the raw embeddings to be projected.
        src_type (int): the type of the source space.
        tgt_type (int): the type of the target space.
        rel_type (int): the type of the relation.
    """
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model = model.to("cuda:0")
    
    model.eval()
    head_type_ids = torch.tensor([src_type] * len(x)).to(x.device)
    rel_type_ids = torch.tensor([rel_type] * len(x)).to(x.device)
    tail_type_ids = torch.tensor([tgt_type] * len(x)).to(x.device)
    output = model(
        head_emb=x,
        head_type_ids=head_type_ids,
        rel_type_ids=rel_type_ids,
        tail_type_ids=tail_type_ids,
    )
    return output['embeddings']


@torch.no_grad()
def project(
    model: BindingModel,
    x: torch.Tensor,
    src_type: int,
    ):
    """Project the raw embeddings to it's space with the modality-specific projection head"""
    if torch.cuda.is_available():
        x = x.to("cuda:0")
        model.to("cuda:0")
    
    model.eval()
    output = model.projection(
        node_emb=x,
        node_type_id=src_type,
    )
    return output


class BridgeInference:
    """Inference based on the trained Bridge model to project raw embeddings to the target space.

    Args:
        model (BindingModel): the trained Bridge model.
    """
    def __init__(self, model: BindingModel):
        self.model = model
        self.model.eval()

    def project(
        self,
        x: torch.Tensor,
        src_type: int,
        batch_size: int = 1024,):
        """Project the raw embeddings to it's space with the modality-specific projection head"""
        # build dataset
        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        all_outputs = []
        for batch in dataloader:
            batch_output = project(
                model=self.model,
                x=batch[0],
                src_type=src_type,
            )
            all_outputs.append(batch_output.cpu())

        if len(all_outputs) > 1:
            all_outputs = torch.cat(all_outputs, dim=0)
        else:
            all_outputs = all_outputs[0]
        return all_outputs

    def transform(self,
        x: torch.Tensor,
        src_type: int,
        tgt_type: int,
        rel_type: int,
        batch_size: int = 1024,
        ):
        """Transform the raw embeddings to the target space.
        """
        # build dataset
        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        # encode
        all_outputs = []
        for batch in dataloader:
            batch_output = transformation(
                model=self.model,
                x=batch[0],
                src_type=src_type,
                tgt_type=tgt_type,
                rel_type=rel_type,
            )
            all_outputs.append(batch_output.cpu())
        
        if len(all_outputs) > 1:
            all_outputs = torch.cat(all_outputs, dim=0)
        else:
            all_outputs = all_outputs[0]
        return all_outputs