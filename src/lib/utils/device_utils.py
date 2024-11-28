import torch


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
    else:
        raise ValueError("Unsupported framework. Choose 'pytorch' or 'keras'.")
