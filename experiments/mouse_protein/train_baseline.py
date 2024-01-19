"""a supervised baseline trained on paired mouse protein and mouse phenotype, taking protein encoder and text encoder.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pdb

from src.text_encoder import load_text_model
from src.text_encoder import inference as text_inference
from src.protein_encoder import load_protein_model
from src.protein_encoder import inference as protein_inference
from src.losses import InfoNCE

import pandas as pd
import transformers
import torch
import torch.nn as nn

import fire

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

class DualEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, df, protein_tokenizer, text_tokenizer):
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer

        # tokenize everything at the beginning
        mpi = df["MP_def"].apply(lambda x: self.text_tokenizer(x, truncation=True))
        self.mpi = mpi.tolist()

        protein = df["Sequence"].tolist()
        self.protein = [self.protein_tokenizer(pro, truncation=True) for pro in protein]

    def __getitem__(self, idx):
        return self.protein[idx], self.mpi[idx]
    
    def __len__(self):
        return len(self.mpi)
    

class TrainCollator:
    def __init__(self, protein_tokenizer, text_tokenizer):
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer

    def __call__(self, batch):
        batch_protein = [x[0] for x in batch]
        batch_text = [x[1] for x in batch]
        batch_protein = self.protein_tokenizer.pad(batch_protein, return_tensors="pt", max_length=512, padding=True, pad_to_multiple_of=8)
        batch_text = self.text_tokenizer.pad(batch_text, return_tensors="pt", max_length=512, padding=True, pad_to_multiple_of=8)
        return batch_protein, batch_text


def main():
    data_folder = "/home/ec2-user/data/mouse_protein/processed/train_test_split"

    # load data
    df_tr = pd.read_csv(os.path.join(data_folder, "triplet_train.csv"))
    df_te = pd.read_csv(os.path.join(data_folder, "triplet_test.csv"))
    protein_model, protein_tokenizer = load_protein_model("facebook/esm2_t33_650M_UR50D")
    text_model, text_tokenizer = load_text_model("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    protein_tokenizer.model_max_length = 512
    text_tokenizer.model_max_length = 512

    # build dataloader
    train_dataset = DualEncoderDataset(df_tr, protein_tokenizer, text_tokenizer)
    test_dataset = DualEncoderDataset(df_te, protein_tokenizer, text_tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=TrainCollator(protein_tokenizer, text_tokenizer))

    # build model
    model = DualEncoderModel(protein_model, text_model)
    model = model.cuda()

    # build training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3):
        iter_loss = 0
        for i,batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch_protein = batch[0]
            batch_text = batch[1]
            batch_protein = {k: v.cuda() for k, v in batch_protein.items()}
            batch_text = {k: v.cuda() for k, v in batch_text.items()}
            loss = model(batch_protein, batch_text)
            loss.backward()
            optimizer.step()
            iter_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss {iter_loss/(i+1)}")
        
    # encode for all protein and text in test set
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=TrainCollator(protein_tokenizer, text_tokenizer))
    protein_emb_list = []
    text_emb_list = []
    model.eval()
    for i,batch in enumerate(test_dataloader):
        batch_protein = batch[0]
        batch_text = batch[1]
        batch_protein = {k: v.cuda() for k, v in batch_protein.items()}
        batch_text = {k: v.cuda() for k, v in batch_text.items()}
        with torch.no_grad():
            pro_emb, text_emb = model(batch_protein, batch_text, return_loss=False)

        protein_emb_list.append(pro_emb.cpu())
        text_emb_list.append(text_emb.cpu())
    
    protein_emb_list = torch.cat(protein_emb_list, dim=0).cpu().numpy()
    text_emb_list = torch.cat(text_emb_list, dim=0).cpu().numpy()

    # save embeddings
    import pickle
    with open("protein_emb.pkl", "wb") as f:
        pickle.dump(protein_emb_list, f)
    
    with open("phenotype_emb.pkl", "wb") as f:
        pickle.dump(text_emb_list, f)

    print("save model checkpoint")
    # torch.save(model.state_dict(), "/home/ec2-user/BioKGBind/experiments/cross_modal_retrieval/checkpoints/mgi_mpo_baseline/model.pt")

    # load model checkpoint
    # model.load_state_dict(torch.load("/home/ec2-user/BioKGBind/experiments/cross_modal_retrieval/checkpoints/mgi_mpo_baseline/model.pt"))
    print("done!")





if __name__ == "__main__":

    fire.Fire(main)
