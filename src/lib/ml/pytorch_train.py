import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR, CosineAnnealingLR

from ..utils.mlflow_setup import log_metrics, log_pytorch_model


def calculate_metrics(y_pred, y_true):
    """
    Calcula métricas relevantes para problemas de regressão:
    MAE, MSE e RMSE.

    Parâmetros:
        y_pred: Tensores com as predições do modelo.
        y_true: Tensores com os valores reais.

    Retorna:
        Um dicionário com as métricas calculadas.
    """
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def train_epoch(model, dataloader, criterion, optimizer=None, is_training=True):
    """
    Treina ou avalia o modelo para uma época.

    Parâmetros:
        model: Modelo a ser treinado ou avaliado.
        dataloader: DataLoader para o conjunto de dados (treinamento ou validação).
        criterion: Função de perda.
        optimizer: Otimizador (apenas para treinamento).
        is_training: Define se é uma etapa de treinamento ou avaliação.

    Retorna:
        A perda média e as métricas de avaliação (MAE, MSE, RMSE).
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

        # Garantir que os tensores estejam no dispositivo correto
        x_batch = x_batch.to(next(model.parameters()).device)
        y_batch = y_batch.to(next(model.parameters()).device)

        # Forward pass
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        if is_training:
            # Backward pass e otimização
            loss.backward()
            optimizer.step()

        # Acumular métricas e perdas
        epoch_loss += loss.item()
        all_y_pred.append(y_pred.detach())
        all_y_true.append(y_batch.detach())

    # Calcular a perda média
    epoch_loss /= len(dataloader)

    # Combinar predições e valores reais para métricas
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
    Treina o modelo para um problema de regressão.

    Parâmetros:
        model: O modelo a ser treinado.
        train_loader: DataLoader para os dados de treino.
        val_loader: DataLoader para os dados de validação.
        epochs: Número de épocas de treinamento.
        lr: Taxa de aprendizado inicial.
        weight_decay: Regularização L2.
        scheduler_type: Tipo de scheduler ('reduce_on_plateau', 'cyclic', 'step', 'cosine_annealing').
        early_stopping_patience: Paciência para early stopping.
        Parâmetros específicos para os schedulers:
            patience, factor, cyclic_base_lr, cyclic_max_lr, step_size_up, step_size, gamma, t_max.

    Retorna:
        O modelo treinado.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Configuração do scheduler
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
            "Tipo de scheduler inválido. Escolha entre 'reduce_on_plateau', 'cyclic', 'step', ou 'cosine_annealing'.")

    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Treinamento
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, is_training=True)

        # Validação
        val_loss, val_metrics = train_epoch(model, val_loader, criterion, is_training=False)

        # Atualizar o scheduler
        if scheduler_type == "reduce_on_plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Log das métricas no MLflow
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

        # Exibição dos resultados
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

    # Restaurar o melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    log_pytorch_model(model)
    return model
