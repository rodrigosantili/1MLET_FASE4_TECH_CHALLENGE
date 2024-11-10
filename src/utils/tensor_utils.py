import torch


def prepare_tensors(data_sequences, data_targets, device):
    """
    Convert data sequences and targets into PyTorch tensors with adjusted dimensions.

    Parameters:
        data_sequences (np.array): Input data sequences.
        data_targets (np.array): Target values corresponding to the sequences.
        device (torch.device): Device to allocate tensors (CPU or GPU).

    Returns:
        X (torch.Tensor): Prepared input tensors.
        y (torch.Tensor): Prepared target tensors.
    """
    X = torch.tensor(data_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
    y = torch.tensor(data_targets, dtype=torch.float32).view(-1, 1).to(device)

    return X, y
