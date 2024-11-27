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

    # Garantir que last_sequence tenha exatamente 3 dimensões: (1, sequence_length, num_features)
    if last_sequence.dim() == 2:  # Caso last_sequence esteja com 2D (seq_length, num_features)
        last_sequence = last_sequence.unsqueeze(0)  # Adicionar dimensão de batch

    with torch.no_grad():
        for _ in range(future_days):
            # Fazer a previsão
            future_pred = model(last_sequence)  # Saída esperada: (batch_size, output_size)

            # Extraindo a previsão
            next_pred = future_pred[0, 0]  # Saída é escalar

            # Armazenar a previsão
            predictions.append(next_pred.item())

            # Ajustar `next_pred` para o formato correto: (1, 1, num_features)
            next_pred_tensor = next_pred.unsqueeze(0).unsqueeze(0).repeat(1, 1, last_sequence.size(-1))

            # Atualizar `last_sequence` para incluir a nova previsão
            last_sequence = torch.cat((last_sequence[:, 1:, :], next_pred_tensor), dim=1)

    # Criar um array de shape (future_days, num_features) para o inverso
    num_features = scaler.scale_.shape[0]
    predictions_array = np.zeros((future_days, num_features))
    predictions_array[:, 0] = predictions  # Preencher apenas a coluna correspondente ao `Close`

    # Transformar as previsões de volta para a escala original
    inverse_transformed = scaler.inverse_transform(predictions_array)
    return inverse_transformed[:, 0]  # Retornar apenas a coluna `Close`
