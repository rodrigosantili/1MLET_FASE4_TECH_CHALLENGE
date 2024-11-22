import numpy as np


def create_sequences(data, seq_length=60):
    """
    Split data into sequences and targets based on a specified sequence length.

    Parameters:
        data (array-like): Data to split into sequences.
        seq_length (int): Length of each sequence.

    Returns:
        sequences (np.array): Data sequences of size seq_length.
        targets (np.array): Next values after each sequence.
    """
    # Garantir que os dados não contenham NaN ou infinito
    assert not np.any(np.isnan(data)), "Os dados contêm valores NaN."
    assert not np.any(np.isinf(data)), "Os dados contêm valores infinitos."

    # Verificar se os dados têm comprimento suficiente
    if len(data) <= seq_length:
        raise ValueError("Os dados devem ter comprimento maior que o comprimento da sequência (seq_length).")

    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        # Extrair sequência e valor-alvo
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])

    return np.array(sequences), np.array(targets)
