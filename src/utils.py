# trainer.py

# Necessary imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import wandb
from dotenv import load_dotenv, dotenv_values

# Dataset classes
class Dataset_txt(Dataset):
    def __init__(self, data, tokenizer, txt_col, target=None, max_length=None, pad_token_id=50256):
        self.data = data
        self.target = target
        self.txt_col = txt_col
        self.encoded_texts = [tokenizer.encode(text) for text in self.data[f"{txt_col}"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        self.encoded_texts = [
            encoded_text[:self.max_length] for encoded_text in self.encoded_texts
        ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

class Dataset_Num(Dataset):
    def __init__(self, data, numerical_features):
        self.data = data
        self.numerical_features = numerical_features
    
    def __getitem__(self, index):
        return (
           torch.tensor(self.data.iloc[index].loc[self.numerical_features], dtype=torch.float),
            torch.tensor(self.data.iloc[index].loc["Epit, mV (SCE)"], dtype=torch.float)
        )
    
    def __len__(self):
        return len(self.data)

# Loss calculation functions
def calc_loss_batch(num_batch, txt_batch, target_batch, model, device):
    num_batch = num_batch.to(device)
    txt_batch = txt_batch.to(device)
    target_batch = target_batch.float().to(device)

    # Forward pass through the model with both numerical and text inputs
    predictions = model(num_batch, txt_batch)

    # Ensure predictions and targets have the same shape
    if predictions.shape != target_batch.shape:
        predictions = predictions.squeeze()

    # Calculate both MAE and MSE losses
    mae_loss = F.l1_loss(predictions, target_batch)
    mse_loss = F.mse_loss(predictions, target_batch)

    return {'mae': mae_loss, 'mse': mse_loss, 'predictions': predictions, 'targets': target_batch}

def calc_loss_loader(data_num_loader, data_txt_loader, model, device, num_batches=None):
    total_mae = 0.
    total_mse = 0.
    all_predictions = []
    all_targets = []

    # Check if both loaders have data
    if len(data_num_loader) == 0 or len(data_txt_loader) == 0:
        return {'mae': float("nan"), 'rmse': float("nan"), 'r2': float("nan")}

    # Determine number of batches to process
    if num_batches is None:
        num_batches = min(len(data_num_loader), len(data_txt_loader))
    else:
        num_batches = min(num_batches, min(len(data_num_loader), len(data_txt_loader)))

    # Create iterators for both loaders
    num_iterator = iter(data_num_loader)
    txt_iterator = iter(data_txt_loader)

    for i in range(num_batches):
        try:
            # Get batches from both loaders
            num_batch, target_batch = next(num_iterator)
            txt_batch = next(txt_iterator)  # Ignore targets from text loader

            batch_results = calc_loss_batch(num_batch, txt_batch, target_batch, model, device)

            total_mae += batch_results['mae'].item()
            total_mse += batch_results['mse'].item()

            # Collect predictions and targets for R² calculation
            all_predictions.append(batch_results['predictions'].cpu().detach())
            all_targets.append(batch_results['targets'].cpu().detach())

        except StopIteration:
            # Handle case where one loader is exhausted before the other
            break

    # Calculate final metrics
    num_processed_batches = len(all_predictions)
    if num_processed_batches == 0:
        return {'mae': float("nan"), 'rmse': float("nan"), 'r2': float("nan")}

    mae = total_mae / num_processed_batches
    rmse = (total_mse / num_processed_batches) ** 0.5

    # Calculate R² score
    all_predictions = torch.cat(all_predictions).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # R² = 1 - (sum of squared residuals / total sum of squares)
    ss_res = ((all_targets - all_predictions) ** 2).sum()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {'mae': mae, 'rmse': rmse, 'r2': r2}

# Main trainer function
def trainer(
    model, train_num_loader, train_txt_loader, val_num_loader, val_txt_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter, loss_function="mae",
    project_name=None, run_name=None):

    # Initialize wandb
    wandb.init(project=project_name, name=run_name)

    # Log hyperparameters
    wandb.config.update({
        "epochs": num_epochs,
        "eval_frequency": eval_freq,
        "eval_iterations": eval_iter,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": optimizer.param_groups['lr'],
        "device": device,
        "model_name": model.__class__.__name__
    })

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_rmses, val_rmses = [], []
    train_r2s, val_r2s = [], []
    examples_seen, global_step = 0, -1

    # Create iterators for the training loaders
    num_train_iter = iter(train_num_loader)
    txt_train_iter = iter(train_txt_loader)

    for epoch in range(num_epochs):
        model.train()
        epoch_mae_loss = 0
        epoch_mse_loss = 0
        batch_count = 0

        # Reset iterators at the beginning of each epoch
        num_train_iter = iter(train_num_loader)
        txt_train_iter = iter(train_txt_loader)

        # Determine number of batches for this epoch
        num_batches = min(len(train_num_loader), len(train_txt_loader))

        for _ in range(num_batches):
            try:
                # Get batches from both loaders
                num_batch, target_batch = next(num_train_iter)
                txt_batch = next(txt_train_iter)  # Ignore targets from text loader

                optimizer.zero_grad()
                loss_dict = calc_loss_batch(
                    num_batch, txt_batch, target_batch, model, device
                )
                # For backward pass, you can choose either MAE or MSE or a combination
                loss = loss_dict[f'{loss_function}']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Track batch-level metrics
                epoch_mae_loss += loss_dict['mae'].item()
                epoch_mse_loss += loss_dict['mse'].item()
                batch_count += 1
                examples_seen += num_batch.shape[0]
                global_step += 1

                # Log batch metrics
                wandb.log({
                    "batch_mae": loss_dict['mae'].item(),
                    "batch_mse": loss_dict['mse'].item(),
                    "batch_rmse": loss_dict['mse'].item() ** 0.5,
                    "examples_seen": examples_seen,
                    "global_step": global_step
                }, step=global_step)

                if global_step % eval_freq == 0:
                    # Evaluate model
                    train_metrics = calc_loss_loader(train_num_loader, train_txt_loader, model, device, eval_iter)
                    val_metrics = calc_loss_loader(val_num_loader, val_txt_loader, model, device, eval_iter)

                    # Store metrics
                    train_losses.append(train_metrics['mae'])
                    val_losses.append(val_metrics['mae'])
                    train_maes.append(train_metrics['mae'])
                    val_maes.append(val_metrics['mae'])
                    train_rmses.append(train_metrics['rmse'])
                    val_rmses.append(val_metrics['rmse'])
                    train_r2s.append(train_metrics['r2'])
                    val_r2s.append(val_metrics['r2'])

                    # Log evaluation metrics
                    wandb.log({
                        "train_mae": train_metrics['mae'],
                        "val_mae": val_metrics['mae'],
                        "train_rmse": train_metrics['rmse'],
                        "val_rmse": val_metrics['rmse'],
                        "train_r2": train_metrics['r2'],
                        "val_r2": val_metrics['r2'],
                        "epoch": epoch + 1
                    }, step=global_step)

                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train MAE {train_metrics['mae']:.3f}, "
                          f"Val MAE {val_metrics['mae']:.3f}, "
                          f"Train R² {train_metrics['r2']:.3f}, "
                          f"Val R² {val_metrics['r2']:.3f}")

            except StopIteration:
                # Handle case where one loader is exhausted before the other
                break

        # Calculate and log metrics at epoch end
        train_metrics = calc_loss_loader(train_num_loader, train_txt_loader, model, device, eval_iter)
        val_metrics = calc_loss_loader(val_num_loader, val_txt_loader, model, device, eval_iter)

        # Log epoch-level metrics
        wandb.log({
            "epoch": epoch + 1,
            "epoch_avg_mae": epoch_mae_loss / batch_count if batch_count > 0 else float('nan'),
            "epoch_avg_mse": epoch_mse_loss / batch_count if batch_count > 0 else float('nan'),
            "epoch_avg_rmse": (epoch_mse_loss / batch_count) ** 0.5 if batch_count > 0 else float('nan'),
            "train_mae": train_metrics['mae'],
            "val_mae": val_metrics['mae'],
            "train_rmse": train_metrics['rmse'],
            "val_rmse": val_metrics['rmse'],
            "train_r2": train_metrics['r2'],
            "val_r2": val_metrics['r2'],
            "learning_rate": optimizer.param_groups['lr']
        }, step=global_step)

        print(f"Training MAE: {train_metrics['mae']:.4f} | ", end="")
        print(f"Validation MAE: {val_metrics['mae']:.4f} | ", end="")
        print(f"Training RMSE: {train_metrics['rmse']:.4f} | ", end="")
        print(f"Validation RMSE: {val_metrics['rmse']:.4f} | ", end="")
        print(f"Training R²: {train_metrics['r2']:.4f} | ", end="")
        print(f"Validation R²: {val_metrics['r2']:.4f}")

    # Close wandb run
    wandb.finish()

    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_maes': train_maes,
        'val_maes': val_maes,
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
        'train_r2s': train_r2s,
        'val_r2s': val_r2s,
        'examples_seen': examples_seen
    }

    return results
