import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR, CosineAnnealingLR
from mlflow_setup import log_metrics, log_pytorch_model


def calculate_accuracy(y_pred, y_true, tolerance=0.05):
    relative_error = torch.abs(y_pred - y_true) / torch.abs(y_true)
    correct = (relative_error < tolerance).float()
    accuracy = correct.mean().item() * 100
    return accuracy


def train_epoch(model, dataloader, criterion, optimizer=None, is_training=False, tolerance=0.05):
    epoch_loss = 0
    epoch_accuracy = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for x_batch, y_batch in dataloader:
        if is_training:
            optimizer.zero_grad()

        x_batch = x_batch.to(next(model.parameters()).device)
        y_batch = y_batch.to(next(model.parameters()).device)

        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += calculate_accuracy(y_pred, y_batch, tolerance)

    epoch_loss /= len(dataloader)
    epoch_accuracy /= len(dataloader)

    return epoch_loss, epoch_accuracy


def train_model(model, train_loader, val_loader, epochs=150, lr=0.001, weight_decay=0.01, tolerance=0.05,
                early_stopping_patience=10, scheduler_type="cyclic", patience=5, factor=0.5,
                cyclic_base_lr=1e-5, cyclic_max_lr=0.0005, step_size_up=10, step_size=30, gamma=0.5, t_max=50):
    """
    Função de treinamento do modelo com a opção de escolher entre diferentes schedulers.

    Parâmetros:
        model: O modelo a ser treinado.
        train_loader: DataLoader para os dados de treino.
        val_loader: DataLoader para os dados de validação.
        epochs: Número de épocas de treinamento.
        lr: Taxa de aprendizado inicial.
        weight_decay: Regularização L2.
        tolerance: Tolerância para calcular a acurácia.
        early_stopping_patience: Paciência para o early stopping.
        scheduler_type: Tipo de scheduler ('reduce_on_plateau', 'cyclic', 'step', 'cosine_annealing').
        patience, factor, cyclic_base_lr, cyclic_max_lr, step_size_up, step_size, gamma, t_max: Parâmetros específicos dos schedulers.
    """

    criterion = nn.MSELoss()  # Função de perda
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Escolher o scheduler com base no parâmetro `scheduler_type`
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

    for epoch in range(epochs):
        # Treinamento
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, is_training=True,
                                                 tolerance=tolerance)

        # Validação
        val_loss, val_accuracy = train_epoch(model, val_loader, criterion, is_training=False, tolerance=tolerance)

        # Atualizar o scheduler
        if scheduler_type == "reduce_on_plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Logar métricas no MLflow
        log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Mostrar progresso a cada 10 épocas
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}% | "
                  f"Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.2f}% | Learning Rate: {current_lr:.6f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    log_pytorch_model(model)
    return model
