import torch
import numpy as np


def future_predictions(model, last_sequence, future_days, scaler):
    """
    Generate future predictions using the trained model.

    Parameters:
        model (torch.nn.Module): The trained LSTM model.
        last_sequence (torch.Tensor): The last sequence of data points to use as input for predictions.
        future_days (int): The number of days to predict into the future.
        scaler (scaler object): The scaler used for inverse transformation.

    Returns:
        np.array: Future predictions, inverse transformed to the original scale.
    """
    predictions = []
    model.eval()

    # Garantir que last_sequence tenha exatamente 3 dimensões: (1, sequence_length, 1)
    last_sequence = last_sequence.view(1, -1, 1)  # (1, sequence_length, 1)

    with torch.no_grad():
        for _ in range(future_days):
            # Fazer a previsão
            future_pred = model(last_sequence)

            # Extrair apenas o último valor do vetor de previsões como a próxima previsão
            next_pred = future_pred[:, -1, 0] if future_pred.dim() == 3 else future_pred[-1, 0]

            # Armazenar a previsão como um escalar
            predictions.append(next_pred.item())

            # Atualizar `last_sequence` para manter o comprimento correto
            # Remover o primeiro elemento e adicionar `next_pred` no final para manter o comprimento constante
            last_sequence = torch.cat((last_sequence[:, 1:, :], next_pred.view(1, 1, 1)), dim=1)

    # Transformar as previsões de volta para a escala original
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))