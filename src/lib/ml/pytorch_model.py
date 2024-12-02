import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=2, dropout=0.2):
        """
        LSTM model to predict time series with multiple features.

        Parameters:
            input_size (int): Number of input features.
            hidden_layer_size (int): Size of the LSTM hidden layer.
            output_size (int): Output size (usually 1 to predict only the price).
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate for regularization.
        """
        super(StockLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # LSTM empilhada com suporte para múltiplas features
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_layer_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)

        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)

        # Camadas densas adicionais
        self.linear_hidden = nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.relu_hidden = nn.ReLU()
        self.linear_post = nn.Linear(hidden_layer_size // 2, output_size)

    def forward(self, input_seq):
        """
        Forward pass of the model.

        Parameters:
            input_seq (tensor): Input tensor with shape (batch_size, seq_length, input_size).

        Returns:
            predictions (tensor): Output tensor with shape (batch_size, output_size).
        """
        # Estados iniciais da LSTM
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(input_seq, (h0, c0))

        # Usar o estado oculto final (última camada)
        combined_hidden = h_n[-1]

        # Aplicar dropout
        combined_hidden = self.dropout(combined_hidden)

        # Passar pelas camadas densas adicionais
        hidden_output = self.relu_hidden(self.linear_hidden(combined_hidden))
        predictions = self.linear_post(hidden_output)

        return predictions
