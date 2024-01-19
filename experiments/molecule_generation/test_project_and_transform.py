import pdb
import os
import json
import pickle
import torch
import pandas as pd
import numpy as np

from src.model import BindingModel
from src.inference import BridgeInference

checkpoint_dir = "./checkpoints/bind-openke-benchmark-6-layer-unimol"
with open(os.path.join(checkpoint_dir, "model_config.json"), "r") as f:
    model_config = json.load(f)
model = BindingModel(**model_config)
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin")))
bind_model = BridgeInference(model)

dis_emb = torch.tensor(np.arange(768),dtype=torch.float32)[None]
dis_emb = bind_model.project(
    x = dis_emb,
    src_type = 2,
)
print(dis_emb[0][:10])
pdb.set_trace()