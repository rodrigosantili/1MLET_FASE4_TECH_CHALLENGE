import torch
import torch.nn as nn

# class StockLSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=2, dropout=0.2):
#         super(StockLSTM, self).__init__()
#         self.hidden_layer_size = hidden_layer_size
#         self.num_layers = num_layers
#
#         # Camada de pré-processamento
#         self.linear_pre = nn.Linear(input_size, hidden_layer_size)
#         self.relu = nn.ReLU()
#
#         # LSTM empilhada com dropout entre as camadas
#         self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#
#         # Apenas o estado oculto final da última camada é usado
#         self.linear_post = nn.Linear(hidden_layer_size, output_size)
#
#     def forward(self, input_seq):
#         # Pré-processamento
#         pre_processed = self.linear_pre(input_seq)
#         pre_processed = self.relu(pre_processed)
#
#         # Estados ocultos iniciais da LSTM
#         h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
#         c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
#
#         # LSTM
#         lstm_out, (h_n, c_n) = self.lstm(pre_processed, (h0, c0))
#
#         # Usar apenas o estado oculto final da última camada
#         combined_hidden = h_n[-1]
#
#         # Dropout aplicado no estado oculto final
#         combined_hidden = self.dropout(combined_hidden)
#
#         # Camada de pós-processamento
#         predictions = self.linear_post(combined_hidden)
#         return predictions

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # Camada de pré-processamento
        self.linear_pre = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()

        # LSTM empilhada com dropout
        self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # Camadas densas adicionais para melhorar a capacidade do modelo
        self.linear_hidden = nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.relu_hidden = nn.ReLU()
        self.linear_post = nn.Linear(hidden_layer_size // 2, output_size)

    def forward(self, input_seq):
        pre_processed = self.linear_pre(input_seq)
        pre_processed = self.relu(pre_processed)

        # Estados ocultos iniciais da LSTM
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(pre_processed, (h0, c0))

        # Usar o estado oculto final
        combined_hidden = h_n[-1]

        # Aplicar dropout
        combined_hidden = self.dropout(combined_hidden)

        # Camadas adicionais
        hidden_output = self.relu_hidden(self.linear_hidden(combined_hidden))
        predictions = self.linear_post(hidden_output)

        return predictions
