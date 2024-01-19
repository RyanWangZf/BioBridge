"""Encode all molecule SMILES strings"""
import pdb
import os
import pickle

import transformers
import pandas as pd
import numpy as np
import torch
import datasets
import fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

from load_model import load_molecule_model

def main(
    # pretrained encoder model name
    model_name = "laituan245/molt5-large-caption2smiles",
    # data path
    data_path = "./data/Processed",
    # output path
    output_path = "./data/embeddings",
    # encode batch size
    batch_size=16,
    ):
    # model device
    device = "cuda:0"

    # tokenizing data path
    tokenized_data_dir=os.path.join(data_path, "./encoded_drug_caption")

    # load model
    model, tokenizer = load_molecule_model(model_name)
    model.to(device)

    if not os.path.exists(tokenized_data_dir):
        def tokenize_data(datapoint):
            # TODO
            pdb.set_trace()
            seq = datapoint['smiles']
            tokenized = tokenizer(
                    seq,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    )            
            return tokenized
        data = load_dataset(data_path, data_files={"drug.csv"})
        data = data["train"].map(tokenize_data, num_proc=1)
        data.save_to_disk(tokenized_data_dir)
    data = datasets.load_from_disk(tokenized_data_dir)
    # save the data node_index, node_id, and node_name
    outputs = {
        "node_index": data["node_index"],
        "node_id": data["drugbank_id"],
        "node_name": data["node_name"],
    }
    remove_columns = []
    for k in data.features.keys():
        if k not in ["input_ids","attention_mask"]:
            remove_columns.append(k)
    data = data.remove_columns(remove_columns)

    loader = DataLoader(data, 
                        batch_size=batch_size,
                        collate_fn=transformers.DataCollatorWithPadding(tokenizer, 
                                max_length=tokenizer.model_max_length,
                                pad_to_multiple_of=8,
                                return_tensors="pt",
                                ),
                        )
    embeddings = []
    for batch in tqdm(loader):
        # map batch components to cuda device
        batch = {k:v.to(device) for k,v in batch.items()}

        with torch.no_grad():
            output = model.get_encoder()(**batch)
        
        emb = output["last_hidden_state"].mean(dim=1)
        emb = emb.cpu().numpy()
        embeddings.append(emb)

    outputs["embedding"] = np.concatenate(embeddings, axis=0)
    # save outputs to disk
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(os.path.join(output_path, "drug_caption.pkl"), "wb") as f:
        pickle.dump(
            outputs,
            f,
        )
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)