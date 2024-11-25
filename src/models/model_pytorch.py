import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=2, dropout=0.2):
        """
        Modelo LSTM para prever séries temporais com múltiplas features.

        Parâmetros:
            input_size (int): Número de features de entrada.
            hidden_layer_size (int): Tamanho da camada oculta da LSTM.
            output_size (int): Tamanho da saída (normalmente 1 para prever apenas o preço).
            num_layers (int): Número de camadas LSTM empilhadas.
            dropout (float): Taxa de dropout para regularização.
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
        Forward pass do modelo.

        Parâmetros:
            input_seq (tensor): Tensor de entrada com shape (batch_size, seq_length, input_size).

        Retorna:
            predictions (tensor): Tensor de saída com shape (batch_size, output_size).
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
