from utils import *

import os
import time
from torch.utils.data import DataLoader

from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

import wandb
wandb.login(key = os.getenv("WANDB_API_KEY"))



data = pd.read_csv(r"..\Code\pit_cleaned_final.csv")
num = ['Fe', 'Cr', 'Ni', 'Mo', 'W', 'N', 'Nb', 'C', 'Si', 'Mn', 'Cu', 'P', 'S',
        'Al', 'V', 'Ta', 'Re', 'Ce', 'Ti', 'Co', 'B', 'Mg', 'Y', 'Gd',
         'Test Temp (C)', '[Cl-] M', 'pH', 'Scan Rate mV/s',
        'Material class']
num_target = ['Epit, mV (SCE)']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[num] = scaler.fit_transform(data[num])

from sklearn.model_selection import train_test_split
train, val = train_test_split(data, test_size=0.1, random_state=42)

train_data = Dataset_txt(
    data = train,
    tokenizer=tokenizer,
    txt_col="combine_text",
    max_length=1024
)
val_data = Dataset_txt(
    data = val,
    tokenizer=tokenizer,
    txt_col="combine_text",
    max_length=1024
)



num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_data,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)


train_num = Dataset_Num(
    data = train,
    numerical_features = num
)
val_num = Dataset_Num(
    data = val,
    numerical_features = num
)
train_num_loader = DataLoader(
    dataset=train_num,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_num_loader = DataLoader(
    dataset=val_num,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)



if __name__:"__main__":
    from model import PitModel
    model = PitModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    start_time = time.time()
    torch.manual_seed(123)

    run_name = "lstm-6-no-txt-grad"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5, verbose=True
    # )
    results = trainer(
        model,train_num_loader, train_loader, val_num_loader, val_loader, optimizer, device,
        num_epochs=5000, eval_freq=10, eval_iter=10,
        project_name="gpt2-corr-txt-num", run_name=run_name
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    torch.save(model.state_dict(), f"{run_name}_model.pt")
    