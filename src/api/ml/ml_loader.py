import logging
import os
import torch

from models.model_pytorch import StockLSTM

MODEL_PATH = "../../res/models/saved/trained_model.pth"
SCALER_PATH = "../../res/models/saved/scaler.pt"


logging.basicConfig(level=logging.INFO)


class MLLoader:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__load_model()
        self.__load_scaler()

    def __load_model(self):
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

            input_size = 4
            hidden_layer_size = 350
            output_size = 1
            num_layers = 3
            dropout = 0.35

            self.model = StockLSTM(
                input_size=input_size,
                hidden_layer_size=hidden_layer_size,
                output_size=output_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(self.device)

            self.model.load_state_dict(torch.load(MODEL_PATH))
            self.model.eval()
            logging.debug(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def __load_scaler(self):
        try:
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

            self.scaler = torch.load(SCALER_PATH)
            logging.debug(f"Scaler loaded from {SCALER_PATH}")
        except Exception as e:
            raise RuntimeError(f"Error loading scaler: {e}")
