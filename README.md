# 1MLET_FASE4_TECH_CHALLENGE
Projeto que desenvolve um modelo LSTM para prever preços de ações. Inclui coleta de dados, criação e treinamento do modelo, deploy de uma API para previsões e monitoramento em produção. Entregáveis: código-fonte, documentação e scripts para deploy.


# Deep Learning Project

This project utilizes a deep learning model implemented with PyTorch to perform data analysis and prediction. The project is organized modularly, with separate components for data collection, preprocessing, model definition, evaluation, and auxiliary functions.

## Project Structure

### General Structure

- **`main.py`**: Main script that executes the entire project pipeline.
- **`mlflow_setup.py`**: Configures MLflow for experiment tracking and logging.

### Directories

#### `data/`
Scripts for data collection and preprocessing.
   - **`fetch_data.py`**: Collects the necessary data for the project.
   - **`preprocess_data.py`**: Processes the collected data, preparing it for the model.

#### `models/`
Scripts for defining, training, and saving the model.
   - **`model_pytorch.py`**: Defines the deep learning model architecture using PyTorch.
   - **`save_model.py`**: Contains the function to save the trained model.
   - **`saved/`**: Stores the saved trained model (`trained_model.pth`).

#### `predict/`
Scripts for making predictions and evaluating model performance.
   - **`evaluate_model.py`**: Evaluates the model's performance with relevant metrics.
   - **`predict_pytorch.py`**: Uses the trained model to make predictions.

#### `utils/`
Auxiliary functions for various tasks across the project.
   - **`device_utils.py`**: Manages device configurations, such as setting up a GPU.
   - **`plot_utils.py`**: Contains functions for data visualization and plotting.
   - **`sequence_utils.py`**: Handles data sequence manipulation and creation.
   - **`tensor_utils.py`**: Utilities for tensor operations and manipulation.

---

## How to Run

1. Install all necessary dependencies:
   ```bash
   pip install -r requirements.txt
2. Configure parameters (`param`) in `main.py` as needed.
3. Execute the main pipeline using:
   ```bash
   python src/main.py
