import torch
import torch.nn as nn
from gpt import *
# loading the orignal model
CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG = {
 "vocab_size": 50257,
 "context_length": 1024,
 "drop_rate": 0.0,
 "qkv_bias": True
}
model_configs = {
 "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
 "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
 "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
 "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

GPT_CONFIG_124M = {
 "vocab_size": 50257,
 "context_length": 256, # We shorten the context length from 1,024 to 256 tokens. Original GPT-2 has a context length of 1,024 tokens.
 "emb_dim": 768,
 "n_heads": 12,
 "n_layers": 12,
 "drop_rate": 0.1,
 "qkv_bias": False
}

from gpt_download import download_and_load_gpt2, load_weights_into_gpt
from gpt import GPTModel
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
)

gpt_model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(gpt_model, params)

gpt_model.eval()
for param in gpt_model.parameters():
    param.requires_grad = False

torch.manual_seed(123)

num_classes = 32 # we take a 32dim output form the gpt model for the representation of text.

gpt_model.out_head = torch.nn.Linear(
 in_features=BASE_CONFIG["emb_dim"],
 out_features=num_classes
)




num = ['Fe', 'Cr', 'Ni', 'Mo', 'W', 'N', 'Nb', 'C', 'Si', 'Mn', 'Cu', 'P', 'S',
        'Al', 'V', 'Ta', 'Re', 'Ce', 'Ti', 'Co', 'B', 'Mg', 'Y', 'Gd',
         'Test Temp (C)', '[Cl-] M', 'pH', 'Scan Rate mV/s',
        'Material class']
num_target = ['Epit, mV (SCE)']

class PitModel(torch.nn.Module):
    def __init__(self, gpt_model = gpt_model,  embedding_dim = 32, numerical_features= len(num)): 
        super().__init__()
        self.num_classes = embedding_dim + numerical_features
        
        self.gpt_model = gpt_model
        # self.tok_emb = nn.Embedding(50257, 256) #vocab size & emb_dim
        # self.pos_emb = nn.Embedding(1024, 256) #context_length & emb_dim
        # self.lstm = nn.LSTM(
        #     input_size=256,
        #     hidden_size=32,
        #     batch_first=True,
        # )
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.num_classes, 256),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x_num, x_text):
        with torch.no_grad():  # Ensure no gradients are calculated for text processing
            x = self.gpt_model(x_text)
            x = x[:, -1, :]
        
            # tok_embeds = self.tok_emb(x_text)
            # pos_embeds = self.pos_emb(
            #     torch.arange(1024, device=x_text.device)
            # )
            # x = tok_embeds + pos_embeds
            # x = self.lstm(x)[0]
            # x= x[:, -1, :]
        
        # Text features are now detached from computation graph
        x = x.detach()
        
        x = torch.cat((x, x_num), dim=1)
        x = self.linear(x)
        return x