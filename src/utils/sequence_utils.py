import numpy as np

def create_sequences(data, seq_length=60, target_column=0):
    """
    Split data into sequences and targets based on a specified sequence length.

    Parameters:
        data (array-like): Data to split into sequences. Assumes multiple features (columns).
        seq_length (int): Length of each sequence.
        target_column (int): Index of the column to use as the target.

    Returns:
        sequences (np.array): Data sequences of shape (num_sequences, seq_length, num_features).
        targets (np.array): Target values of shape (num_sequences,).
    """
    # Garantir que os dados não contenham NaN ou infinito
    assert not np.any(np.isnan(data)), "Os dados contêm valores NaN."
    assert not np.any(np.isinf(data)), "Os dados contêm valores infinitos."

    # Verificar se os dados têm comprimento suficiente
    if len(data) <= seq_length:
        raise ValueError("Os dados devem ter comprimento maior que o comprimento da sequência (seq_length).")

    # Garantir que a coluna-alvo exista
    if target_column >= data.shape[1]:
        raise ValueError(f"O índice da coluna-alvo ({target_column}) é maior que o número de colunas ({data.shape[1]}).")

    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        # Extrair sequência de todas as features
        seq = data[i:i + seq_length]
        # Extrair o valor alvo apenas da coluna especificada
        target = data[i + seq_length, target_column]

        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)

