import torch


def save_model_local(model, path='model.pth'):
    """
    Salva o modelo treinado localmente no formato PyTorch.
    """
    torch.save(model.state_dict(), path)
    print(f"Modelo salvo em {path}")


def save_scaler_torch(scaler, path):
    """
    Salva o scaler usando torch.save.

    Parameters:
        scaler: O objeto scaler treinado (e.g., MinMaxScaler).
        path (str): Caminho onde o scaler será salvo.
    """
    torch.save(scaler, path)
    print(f"Scaler salvo em {path}")
