import torch


def prepare_tensors_pytorch(data_sequences, data_targets, device):
    """
    Convert data sequences and targets into PyTorch tensors with the correct dimensions.

    Parameters:
        data_sequences (np.array): Input data sequences.
        data_targets (np.array): Target values corresponding to the sequences.
        device (torch.device): Device to allocate tensors (CPU or GPU).

    Returns:
        X (torch.Tensor): Prepared input tensors with 3 dimensions (batch_size, sequence_length, input_size).
        y (torch.Tensor): Prepared target tensors with 2 dimensions (batch_size, target_size).
    """
    # Ensure that X has 3 dimensions: (batch_size, sequence_length, input_size)
    X = torch.tensor(data_sequences, dtype=torch.float32).to(device)

    # Ensure that y has 2 dimensions: (batch_size, target_size)
    y = torch.tensor(data_targets, dtype=torch.float32).view(-1, 1).to(device)

    return X, y
