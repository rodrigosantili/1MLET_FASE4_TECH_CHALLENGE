import os

import torch


def save_model_local(model, path='model.pth'):
    """
    Saves the trained model locally in PyTorch format.

    Parameters:
        model (torch.nn.Module): The trained PyTorch model to be saved.
        path (str): The file path where the model will be saved. Default is 'model.pth'.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Modelo salvo em {path}")


def save_scaler_torch(scaler, path):
    """
    Saves the scaler using torch.save.

    Parameters:
        scaler: The trained scaler object (e.g., MinMaxScaler).
        path (str): The file path where the scaler will be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(scaler, path)
    print(f"Scaler salvo em {path}")
