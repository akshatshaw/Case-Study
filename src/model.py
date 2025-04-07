import torch
import torch.nn as nn

num = ['Fe', 'Cr', 'Ni', 'Mo', 'W', 'N', 'Nb', 'C', 'Si', 'Mn', 'Cu', 'P', 'S',
        'Al', 'V', 'Ta', 'Re', 'Ce', 'Ti', 'Co', 'B', 'Mg', 'Y', 'Gd',
         'Test Temp (C)', '[Cl-] M', 'pH', 'Scan Rate mV/s',
        'Material class']
num_target = ['Epit, mV (SCE)']

class PitModel(torch.nn.Module):
    def __init__(self,  embedding_dim = 32, numerical_features= len(num)): #gpt_model,
        super().__init__()
        self.num_classes = embedding_dim + numerical_features
        
        # self.gpt_model = gpt_model
        self.tok_emb = nn.Embedding(50257, 256) #vocab size & emb_dim
        self.pos_emb = nn.Embedding(1024, 256) #context_length & emb_dim
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=32,
            batch_first=True,
        )
        
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
        # x = self.gpt_model(x_text)
        # x = x[:, -1, :]
        # x = torch.cat((x, x_num), dim=1)
        
        with torch.no_grad():  # Ensure no gradients are calculated for text processing
            tok_embeds = self.tok_emb(x_text)
            pos_embeds = self.pos_emb(
                torch.arange(1024, device=x_text.device)
            )
            x = tok_embeds + pos_embeds
            x = self.lstm(x)[0]
            x= x[:, -1, :]
        
        # Text features are now detached from computation graph
        x = x.detach()
        
        x = torch.cat((x, x_num), dim=1)
        x = self.linear(x)
        return x