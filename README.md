# 1MLET_FASE4_TECH_CHALLENGE
Project that develops an LSTM model to predict Bitcoin prices. It includes data collection, model creation and training, deployment of an API for predictions, and monitoring in production.

# Deep Learning Project

This project develops an LSTM model to predict Bitcoin prices. It includes data collection, model creation and training with PyTorch, deployment of an API for predictions, experiment tracking and model management with MLflow. 
The API stack built using FastAPI for the web framework, Docker for containerization, Grafana and Prometheus for monitoring.
The API provides and endpoints for making predictions and monitoring performance. 
The project is organized modularly, with separate components for data collection, preprocessing, model definition, evaluation, and auxiliary functions.


## Project Structure

### General Structure

- **`/src`**: Contains all the source code for the project, including scripts for data collection, preprocessing, model training, and API services.
- **`/res`**: Stores resources such as configuration files, datasets, and other static files required by the project.
- **`/grafana`**: Includes configuration files and dashboards for monitoring the model's performance and other metrics using Grafana.

### Directories

#### `src/api`
Contains scripts and modules for setting up and running the API service. This includes the implementation of endpoints, request handling, and integration with the prediction model.

### `src/lib`
Houses utility libraries and helper functions that are used across different parts of the project. This can include custom modules for data processing, logging, configuration management, and other reusable components.

### `src/training`
Includes scripts and resources for training the machine learning model. This directory typically contains code for data preprocessing, model training, validation, and saving the trained model.

---

## How to Run

### Model Training and Monitoring
1. Install all necessary dependencies:
   ```bash
   pip install -r requirements.txt
2. Configure parameters (`param`) in `src/training/main.py` as needed.
3. Execute the main pipeline using:
   ```bash
   python -m src.training.main

### API
1. Install all necessary dependencies:
   ```bash
   pip install -r requirements.txt
2. Run the docker-compose file to start the API service:
   ```bash
   docker-compose up --build
