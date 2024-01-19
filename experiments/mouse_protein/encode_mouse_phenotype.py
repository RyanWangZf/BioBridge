# encode all phenotypes in the mouse protein dataset

"""
"""
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

from src.text_encoder import load_text_model, inference

def main(
    # pretrained encoder model name
    model_name = "dmis-lab/biobert-base-cased-v1.2",
    # data path
    data_path = "./data/Processed",
    # output path
    output_path = "./data/embeddings",
    # encode batch size
    batch_size=64,
    ):
    # model device
    device = "cuda:0"

    # load model
    model, tokenizer = load_text_model(model_name)
    model.to(device)

    tokenizer.model_max_length = 512

    # tokenizing data path
    tokenized_data_dir = os.path.join(data_path, "./encoded_mp")

    # load data
    if not os.path.exists(tokenized_data_dir):
        def tokenize_data(datapoint):
            seq = "Name: {}. Definition: {}".format(datapoint["name"], datapoint["def"])
            tokenized = tokenizer(
                    seq,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    max_length=512,
                    )
            tokenized.update(datapoint)
            return tokenized
        data = load_dataset("csv", data_files=os.path.join(data_path, "mouse_phenotype.csv"))
        data = data["train"].map(tokenize_data, num_proc=8)
        data.save_to_disk(tokenized_data_dir)
    data = datasets.load_from_disk(tokenized_data_dir)

    # save the data node_index, node_id, and node_name
    outputs = {"mpi": data["id"], "name": data["name"]}
    data = data.remove_columns(["synonyms","id","name","is_obsolete","def"])

    # start encoding using protein model
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
        emb = inference(model, batch)
        emb = emb.cpu().numpy()
        embeddings.append(emb)

    outputs["embedding"] = np.concatenate(embeddings, axis=0)

    # save outputs to disk
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(os.path.join(output_path, "mp.pkl"), "wb") as f:
        pickle.dump(
            outputs,
            f,
        )
    print("Done!")

if __name__ == "__main__":
    fire.Fire(main)