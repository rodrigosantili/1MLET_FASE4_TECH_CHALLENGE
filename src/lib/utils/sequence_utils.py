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
    # Ensure the data does not contain NaN or infinite values
    assert not np.any(np.isnan(data)), "The data contains NaN values."
    assert not np.any(np.isinf(data)), "The data contains infinite values."

    # Check if the data has sufficient length
    if len(data) <= seq_length:
        raise ValueError("The data must be longer than the sequence length (seq_length).")

    # Ensure the target column exists
    if target_column >= data.shape[1]:
        raise ValueError(f"The target column index ({target_column}) is greater than the number of columns ({data.shape[1]}).")

    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        # Extract sequence of all features
        seq = data[i:i + seq_length]
        # Extract the target value only from the specified column
        target = data[i + seq_length, target_column]

        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)
