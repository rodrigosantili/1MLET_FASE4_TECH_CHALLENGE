import logging

import tensorflow as tf
import joblib
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "lstm_model.keras")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.save")


logging.basicConfig(level=logging.INFO)


class MLLoader:
    def __init__(self):
        self.__model = None
        self.__scaler = None
        self.__load_model()
        self.__load_scaler()

    def __load_model(self):
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

            self.model = tf.keras.models.load_model(MODEL_PATH)
            logging.debug(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def __load_scaler(self):
        try:
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

            self.scaler = joblib.load(SCALER_PATH)
            logging.debug(f"Scaler loaded from {SCALER_PATH}")
        except Exception as e:
            raise RuntimeError(f"Error loading scaler: {e}")
