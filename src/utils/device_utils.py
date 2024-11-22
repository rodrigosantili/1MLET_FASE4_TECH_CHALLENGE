import torch
# import tensorflow as tf


def get_device(framework="pytorch"):
    """
    Returns the appropriate device for PyTorch or Keras/TensorFlow.

    Parameters:
        framework (str): The framework being used, either "pytorch" or "keras".

    Returns:
        device (torch.device or str): The device to be used, either "cuda" or "cpu" for PyTorch,
                                      or "/GPU:0" or "/CPU:0" for Keras/TensorFlow.
    """
    if framework == "pytorch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for PyTorch: {device}")
        return device
    # elif framework == "keras":
    #     device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    #     print(f"Using device for Keras: {device}")
    #     return device
    else:
        raise ValueError("Unsupported framework. Choose 'pytorch' or 'keras'.")
