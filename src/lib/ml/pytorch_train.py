import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR, CosineAnnealingLR

from ..utils.mlflow_setup import log_metrics, log_pytorch_model


def calculate_metrics(y_pred, y_true):
    """
    Calculates relevant metrics for regression problems: MAE, MSE, and RMSE.

    Parameters:
        y_pred (torch.Tensor): Tensors with the model predictions.
        y_true (torch.Tensor): Tensors with the actual values.

    Returns:
        dict: A dictionary with the calculated metrics.
    """
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def train_epoch(model, dataloader, criterion, optimizer=None, is_training=True):
    """
    Trains or evaluates the model for one epoch.

    Parameters:
        model (torch.nn.Module): The model to be trained or evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset (training or validation).
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer, optional): Optimizer (only for training).
        is_training (bool): Indicates whether it is a training or evaluation step.

    Returns:
        tuple: The average loss and evaluation metrics (MAE, MSE, RMSE).
    """
    epoch_loss = 0
    all_y_pred = []
    all_y_true = []

    if is_training:
        model.train()
    else:
        model.eval()

    for x_batch, y_batch in dataloader:
        if is_training:
            optimizer.zero_grad()

        # Ensure tensors are on the correct device
        x_batch = x_batch.to(next(model.parameters()).device)
        y_batch = y_batch.to(next(model.parameters()).device)

        # Forward pass
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        if is_training:
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Accumulate metrics and losses
        epoch_loss += loss.item()
        all_y_pred.append(y_pred.detach())
        all_y_true.append(y_batch.detach())

    # Calculate average loss
    epoch_loss /= len(dataloader)

    # Combine predictions and actual values for metrics
    all_y_pred = torch.cat(all_y_pred)
    all_y_true = torch.cat(all_y_true)

    metrics = calculate_metrics(all_y_pred, all_y_true)
    return epoch_loss, metrics


def train_model(
    model, train_loader, val_loader, epochs=150, lr=0.001, weight_decay=0.01,
    scheduler_type="step", early_stopping_patience=10,
    patience=5, factor=0.5, cyclic_base_lr=1e-5, cyclic_max_lr=0.0005,
    step_size_up=10, step_size=30, gamma=0.5, t_max=50
):
    """
    Trains the model for a regression problem.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        epochs (int): Number of training epochs.
        lr (float): Initial learning rate.
        weight_decay (float): L2 regularization.
        scheduler_type (str): Type of scheduler ('reduce_on_plateau', 'cyclic', 'step', 'cosine_annealing').
        early_stopping_patience (int): Patience for early stopping.
        patience (int): Patience for the scheduler.
        factor (float): Factor for the scheduler.
        cyclic_base_lr (float): Base learning rate for cyclic scheduler.
        cyclic_max_lr (float): Maximum learning rate for cyclic scheduler.
        step_size_up (int): Step size up for cyclic scheduler.
        step_size (int): Step size for step scheduler.
        gamma (float): Gamma for step scheduler.
        t_max (int): Maximum number of iterations for cosine annealing scheduler.

    Returns:
        torch.nn.Module: The trained model.
    """
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler configuration
    if scheduler_type == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
    elif scheduler_type == "cyclic":
        scheduler = CyclicLR(optimizer, base_lr=cyclic_base_lr, max_lr=cyclic_max_lr, step_size_up=step_size_up,
                             mode='triangular2')
    elif scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "cosine_annealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
    else:
        raise ValueError(
            "Invalid scheduler type. Choose between 'reduce_on_plateau', 'cyclic', 'step', or 'cosine_annealing'.")

    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, is_training=True)

        # Validation
        val_loss, val_metrics = train_epoch(model, val_loader, criterion, is_training=False)

        # Update scheduler
        if scheduler_type == "reduce_on_plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Log metrics to MLflow
        log_metrics({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_MAE": train_metrics["MAE"],
            "train_MSE": train_metrics["MSE"],
            "train_RMSE": train_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
            "val_MSE": val_metrics["MSE"],
            "val_RMSE": val_metrics["RMSE"],
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Display results
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Train RMSE: {train_metrics['RMSE']:.4f} | Val RMSE: {val_metrics['RMSE']:.4f} | "
                  f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    log_pytorch_model(model)
    return model
