import os

import torch


def save_model_local(model, path='model.pth'):
    """
    Salva o modelo treinado localmente no formato PyTorch.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Modelo salvo em {path}")


def save_scaler_torch(scaler, path):
    """
    Salva o scaler usando torch.save.

    Parameters:
        scaler: O objeto scaler treinado (e.g., MinMaxScaler).
        path (str): Caminho onde o scaler será salvo.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(scaler, path)
    print(f"Scaler salvo em {path}")
